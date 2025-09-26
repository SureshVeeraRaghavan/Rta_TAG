import os
import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
import json
from flask import Flask
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ===============================
# 1. LOAD DATA
# ===============================
DATA_FOLDER = "./dataset"

file_path = os.path.join(DATA_FOLDER, "ridership.csv")
df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

print(f"CSV loaded successfully. Total rows: {len(df)}")
# ===============================
# 2. EMBEDDINGS + VECTOR DB
# ===============================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma client
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("ridership_data")

documents, metadatas, ids = [], [], []
for idx, row in df.iterrows():
    date_str = row['date'].strftime('%A, %B %d, %Y')
    ridership_details = [
        f"{col.replace('_', ' ')} had {int(row[col])} trips"
        for col in df.columns if col != "date" and pd.notna(row[col])
    ]
    atomic_chunk = f"On {date_str}, " + "; ".join(ridership_details) + "."

    documents.append(atomic_chunk)
    metadatas.append({
        "row_id": idx,
        "date": row['date'].isoformat()
    })
    ids.append(str(idx))

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
    embeddings=embedding_model.encode(documents).tolist()
)

print(f"Chroma collection created with {len(documents)} chunks.")

# Reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# Gemini setup
genai.configure(api_key="AIzaSyAfaUbwr-vzlUbt66znKxSnbOqgESd7moE")  # replace with your Gemini API key
synthesis_model = genai.GenerativeModel("gemini-2.5-pro")# ===============================
# 3. RETRIEVAL
# ===============================
def vector_retrieval(query, top_k=10):
    query_emb = embedding_model.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    if not results["documents"]:
        return []
    docs = results["documents"][0]
    scores = results["distances"][0]

    rerank_pairs = [[query, doc] for doc in docs]
    rerank_scores = reranker.predict(rerank_pairs)

    reranked = sorted(zip(docs, rerank_scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:top_k]]# ===============================
# 4. QUERY CLASSIFIER (UPGRADED)
# ===============================
def classify_query(query):
    """Detect whether query is aggregation (math) or descriptive."""
    agg_keywords = [
        "total", "sum", "average", "mean", "max", "min", "highest", "lowest",
        "busiest", "least", "compare", "vs"
    ]
    if any(word in query.lower() for word in agg_keywords):
        return "aggregation"
    return "descriptive"
# ===============================
# 5. HYBRID ANSWER ENGINE (UPGRADED)
# ===============================
def answer_query(query):
    query_type = classify_query(query)
    print(f"🔍 Detected query type: {query_type}")

    df_copy = df.copy()

    import re
    from datetime import datetime
    import pandas as pd

    # Use the full, original list of columns for ranking later
    original_transport_cols = [col for col in df_copy.columns if col != "date" and col.lower() != "week"]
    transport_cols = original_transport_cols.copy()
    ql = query.lower().strip()

    comparison_split_match = re.search(r'compare\s+(.+?)\s+(?:with|to|vs\.?|versus)\s+(.+)', ql, re.IGNORECASE)

    if ("all modes" in ql or "all transport modes" in ql) and comparison_split_match:
        df_copy['All Transport Modes'] = df_copy[transport_cols].sum(axis=1)
        transport_cols = ['All Transport Modes']
        df_copy = df_copy[['date', 'All Transport Modes']]
        matched_cols = ['All Transport Modes']
    else:
        if "all modes" in ql or "all transport modes" in ql:
            matched_cols = transport_cols
        else:
            matched_cols = [col for col in transport_cols if col.lower() in query.lower()]
        
        # ========== NEW FIX STARTS HERE ==========
        # If no specific transport mode is mentioned in an aggregation query, default to the main 'Metro' column.
        # This handles queries like "what was the average for the first 10 days of march?".
        if not matched_cols and query_type == "aggregation":
            print("🤔 No specific transport mode found in query, defaulting to 'Metro' ridership.")
            matched_cols = ['Metro']
        # ========== NEW FIX ENDS HERE ==========


    def _month_number(name):
        months = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}
        return months.get(name.lower())

    tmap = {c.lower(): c for c in transport_cols}

    def _slice_summary(slice_df, col):
        sdf = slice_df.copy()
        if not sdf.empty: sdf["week_of_month"] = sdf["date"].apply(lambda d: (d.day - 1) // 7 + 1)
        total = sdf[col].sum() if not sdf.empty else 0
        avg = sdf[col].mean() if not sdf.empty else 0
        if not sdf.empty:
            max_row, min_row = sdf.loc[sdf[col].idxmax()], sdf.loc[sdf[col].idxmin()]
            busiest_week_series = sdf.groupby("week_of_month")[col].sum()
            busiest_week = busiest_week_series.idxmax() if not busiest_week_series.empty else "N/A"
            busiest_week_trips = busiest_week_series.max() if not busiest_week_series.empty else 0
            max_date, min_date = max_row["date"].date(), min_row["date"].date()
            max_val, min_val = int(max_row[col]), int(min_row[col])
        else:
            busiest_week, busiest_week_trips, max_date, min_date, max_val, min_val = "N/A", 0, None, None, 0, 0
        return {"total": int(total), "avg": float(avg) if not pd.isna(avg) else 0.0, "busiest_day_date": max_date, "busiest_day_value": max_val, "quietest_day_date": min_date, "quietest_day_value": min_val, "busiest_week": busiest_week, "busiest_week_trips": int(busiest_week_trips)}

    def _parse_period(text_chunk):
        text_chunk = text_chunk.strip()
        week_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s+week\s+of\s+(\w+)(?:\s+(\d{4}))?', text_chunk, re.IGNORECASE)
        if week_match:
            wk_num, mon_str, year_str = week_match.groups()
            mon_num = _month_number(mon_str)
            if not mon_num: return None
            def get_ordinal(n): return str(n) + ("th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))
            label = f"{get_ordinal(int(wk_num))} Week of {mon_str.title()}"
            if year_str: label += f" {year_str}"
            return {"type": "week", "week": int(wk_num), "month": mon_num, "year": int(year_str) if year_str else None, "label": label}
        day_range_match = re.search(r'(first|last)\s+(\d+)\s+days(?:\s+of\s+(\w+))?', text_chunk, re.IGNORECASE)
        if day_range_match:
            direction, num_days_str, month_str = day_range_match.groups()
            month_num = _month_number(month_str) if month_str else None
            label = f"{direction.title()} {num_days_str} Days"
            if month_str: label += f" of {month_str.title()}"
            return {"type": "day_range", "direction": direction, "num_days": int(num_days_str), "month": month_num, "label": label}
        month_match = re.search(r'(?:month\s+of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?', text_chunk, re.IGNORECASE)
        if month_match:
            mon_str, year_str = month_match.groups()
            mon_num = _month_number(mon_str)
            if not mon_num: return None
            label = f"{mon_str.title()}"
            if year_str: label += f" {year_str}"
            else: label = f"The month of {label}"
            return {"type": "month", "month": mon_num, "year": int(year_str) if year_str else None, "label": label}
        year_match = re.search(r'(?:year\s+)?(\d{4})', text_chunk, re.IGNORECASE)
        if year_match:
            year_str = year_match.group(1)
            return {"type": "year", "year": int(year_str), "label": f"The Year {year_str}"}
        return None

    def _get_data_for_period(period):
        target_df = df_copy.copy()
        if period.get("year"): target_df = target_df[target_df['date'].dt.year == period["year"]]
        if period.get("month"): target_df = target_df[target_df['date'].dt.month == period["month"]]
        if period.get("month") == 5:
            end_date_may = pd.to_datetime('2025-05-11')
            target_df = target_df[target_df['date'] <= end_date_may]
        if period["type"] == "week":
            if not target_df.empty:
                target_df['week_of_month'] = target_df['date'].apply(lambda d: (d.day - 1) // 7 + 1)
                return target_df[target_df['week_of_month'] == period["week"]].copy()
            return pd.DataFrame()
        elif period["type"] == "day_range":
            if not target_df.empty:
                if period["direction"] == 'first': return target_df.nsmallest(period["num_days"], 'date').copy()
                else: return target_df.nlargest(period["num_days"], 'date').copy()
            return pd.DataFrame()
        return target_df

    month_names_list = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    found_months = sorted([m for m in month_names_list if m in ql], key=lambda m: month_names_list.index(m))

    # --- ROUTE 1: Multi-Period Comparison ---
    day_range_slice_match = re.search(r'(first|last)\s+(\d+)\s+days', ql, re.IGNORECASE)
    if "compare" in ql and len(found_months) >= 2 and not comparison_split_match:
        print(f"📊 Performing multi-period comparison for months: {', '.join(found_months)}")
        if not matched_cols: return "❌ Please specify a transport mode to compare."
        col_to_compare = matched_cols[0]
        summaries, labels = [], []
        for month_name in found_months:
            month_num = _month_number(month_name)
            
            if day_range_slice_match:
                direction, num_days_str = day_range_slice_match.groups()
                period = {
                    'type': 'day_range', 'direction': direction, 'num_days': int(num_days_str), 'month': month_num,
                    'label': f'{direction.title()} {num_days_str} Days of {month_name.title()}'
                }
            else:
                period = {'type': 'month', 'month': month_num, 'label': month_name.title()}
            
            df_period = _get_data_for_period(period)
            if not df_period.empty:
                summaries.append(_slice_summary(df_period, col_to_compare))
                labels.append(period['label'])

        if not summaries: return f"❌ No data available for the specified months."
        import matplotlib.pyplot as plt; import textwrap
        metric_key, ylabel = ("Total Trips", "Total Trips")
        if any(k in ql for k in ["average", "avg", "mean"]): metric_key, ylabel = ("Average Daily Trips", "Average Daily Trips")
        metric_dict_key = 'total' if metric_key == "Total Trips" else 'avg'
        plot_values = [s[metric_dict_key] for s in summaries]
        plt.style.use('seaborn-v0_8-darkgrid'); fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
        bars = ax.bar(labels, plot_values, color=colors[:len(labels)])
        ax.set_ylabel(ylabel, fontsize=14); ax.set_title(f"{metric_key} for {col_to_compare.replace('_', ' ').title()}", fontsize=16, fontweight="bold")
        ymax = max(plot_values) if plot_values else 0
        for bar, val in zip(bars, plot_values):
            disp = f"{int(val):,}" if metric_key != "Average Daily Trips" else f"{val:,.2f}"
            ax.text(bar.get_x() + bar.get_width()/2, val + (ymax*0.01 if ymax else 0.05), disp, ha="center", va="bottom", fontsize=11, fontweight="bold")
        plt.tight_layout(); plt.show()
        context_lines = [f"For {label}: total trips = {summary['total']:,}, average daily trips = {summary['avg']:,.2f}." for label, summary in zip(labels, summaries)]
        context = "\n".join(context_lines)
        prompt = f"You are a ridership analysis assistant. Based on the following numeric context, generate a full, natural-language comparison for the user query.\n\nCONTEXT:\n{context}\n\nUSER QUERY:\n{query}\n\nANSWER:"
        response = synthesis_model.generate_content(prompt); return response.text

    # --- ROUTE 2: Binary Comparison ---
    elif query_type == "aggregation" and matched_cols and comparison_split_match:
        part1_str, part2_str = comparison_split_match.group(1).strip(), comparison_split_match.group(2).strip()
        modes1, modes2 = [c for c in transport_cols if c.lower() in part1_str.lower()], [c for c in transport_cols if c.lower() in part2_str.lower()]
        if "all transport modes" in ql or "all modes" in ql: modes1 = modes2 = ['All Transport Modes']
        elif modes1 and not modes2: modes2 = modes1
        elif not modes1 and matched_cols:
            modes1 = [matched_cols[0]]
            if not modes2: modes2 = [matched_cols[0]]
        if not modes1 or not modes2: return "❌ Could not identify a specific transport mode for each side of the comparison."
        col1, col2 = modes1[0], modes2[0]
        period1, period2 = _parse_period(part1_str), _parse_period(part2_str)
        if period1 and period2:
            if (period2['type'] == 'day_range' and period2.get('month') is None and period1.get('month') is not None):
                period2['month'] = period1['month']
                reverse_months = { 1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December" }
                month_name = reverse_months.get(period1['month'])
                if month_name: period2['label'] += f" of {month_name}"
            print(f"📊 Performing comparison: '{col1} in {period1['label']}' vs '{col2} in {period2['label']}'")
            df1, df2 = _get_data_for_period(period1), _get_data_for_period(period2)
            if df1.empty or df2.empty: return f"❌ Could not find data for one or both periods ('{period1['label']}', '{period2['label']}'). Please check your query."
            s1, s2 = _slice_summary(df1, col1), _slice_summary(df2, col2)
            import matplotlib.pyplot as plt; import textwrap
            label1, label2 = f"{col1.replace('_', ' ').title()} ({period1['label']})", f"{col2.replace('_', ' ').title()} ({period2['label']})"
            plot_df = pd.DataFrame({"Period": [label1, label2], "Total Trips": [s1["total"], s2["total"]], "Average Daily Trips": [s1["avg"], s2["avg"]]})
            if not plot_df.empty:
                plt.style.use('seaborn-v0_8-darkgrid'); fig, ax = plt.subplots(figsize=(10, 6))
                labels = [textwrap.fill(label, 20) for label in plot_df["Period"]]
                metric_key, ylabel = "Total Trips", "Total Trips"
                if any(k in ql for k in ["average", "avg", "mean"]): metric_key, ylabel = "Average Daily Trips", "Average Daily Trips"
                values = plot_df[metric_key]; bars = ax.bar(labels, values, color=['#3B82F6', '#10B981'])
                ax.set_ylabel(ylabel, fontsize=14); ax.set_title("Ridership Comparison", fontsize=16, fontweight="bold")
                ymax = max(values) if len(values) > 0 else 0
                for bar, val in zip(bars, values):
                    disp = f"{int(val):,}" if metric_key != "Average Daily Trips" else f"{val:,.2f}"
                    ax.text(bar.get_x() + bar.get_width()/2, val + (ymax*0.01 if ymax else 0.05), disp, ha="center", va="bottom", fontsize=11, fontweight="bold")
                plt.tight_layout(); plt.show()
            context1, context2 = f"For {label1}: total trips = {s1['total']:,}, average daily trips = {s1['avg']:,.2f}.", f"For {label2}: total trips = {s2['total']:,}, average daily trips = {s2['avg']:,.2f}."
            context = f"{context1}\n{context2}"
            prompt = f"You are a ridership analysis assistant. Based on the following numeric context, generate a full, natural-language comparison for the user query.\n\nCONTEXT:\n{context}\n\nUSER QUERY:\n{query}\n\nANSWER:"
            response = synthesis_model.generate_content(prompt); return response.text

    # --- ROUTE 3: Single Period Aggregation ---
    elif query_type == "aggregation" and matched_cols:
        filtered_df = df_copy.copy(); range_match = re.search(r'(\w+)\s+(\d{1,2}).*?(?:to|through|-|and)\s*(\w+)?\s*(\d{1,2})', ql)
        first_last_match = re.search(r'(first|last)\s+(\d+)\s+days(?:\s+of\s+(\w+))?', ql); day_match = re.search(r'(?:on|for|in|of)?\s*(?:(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(\w+)|(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?)', ql)
        week_matches = re.findall(r'(\d+)(?:st|nd|rd|th)? week of (\w+)', ql); date_filter_applied = False
        try:
            if range_match:
                month1_name, day1_str, month2_name, day2_str = range_match.groups(); month1_num, day1 = _month_number(month1_name), int(day1_str)
                month2_num, day2 = _month_number(month2_name) if month2_name else month1_num, int(day2_str)
                if month1_num and month2_num:
                    year = df_copy['date'].dt.year.max(); start_date, end_date = pd.to_datetime(f'{year}-{month1_num}-{day1}'), pd.to_datetime(f'{year}-{month2_num}-{day2}')
                    if start_date > end_date: end_date = pd.to_datetime(f'{year+1}-{month2_num}-{day2}')
                    filtered_df = df_copy[(df_copy['date'] >= start_date) & (df_copy['date'] <= end_date)].copy()
                    if not filtered_df.empty: date_filter_applied = True; print(f"🗓️ Applied date range filter: {start_date.date()} to {end_date.date()}")
            elif first_last_match:
                direction, num_days_str, month_name = first_last_match.groups(); num_days, target_df = int(num_days_str), df_copy.copy()
                if month_name:
                    month_num = _month_number(month_name)
                    if month_num: target_df = df_copy[df_copy['date'].dt.month == month_num].copy()
                if not target_df.empty:
                    if direction == 'first': filtered_df = target_df.nsmallest(num_days, 'date').copy()
                    else: filtered_df = target_df.nlargest(num_days, 'date').copy()
                    if not filtered_df.empty: date_filter_applied = True; print(f"🗓️ Applied filter: {direction.title()} {num_days} days" + (f" of {month_name.title()}" if month_name else ""))
            elif day_match:
                day_g1, month_g1, month_g2, day_g2 = day_match.groups(); day_num, month_num, month_str_print = (None, None, None)
                if day_g1 and month_g1: day_num, month_num, month_str_print = int(day_g1), _month_number(month_g1), month_g1
                elif month_g2 and day_g2: day_num, month_num, month_str_print = int(day_g2), _month_number(month_g2), month_g2
                if month_num and day_num:
                    filtered_df = df_copy[(df_copy['date'].dt.month == month_num) & (df_copy['date'].dt.day == day_num)].copy()
                    if not filtered_df.empty: date_filter_applied = True; print(f"🗓️ Applied single day filter: {month_str_print.title()} {day_num}")
        except (ValueError, TypeError, IndexError) as e: print(f"⚠️ Could not parse specific date from query: {e}. Proceeding with general logic."); date_filter_applied = False
        if len(week_matches) == 1 and not date_filter_applied:
            (wk_num, wk_month) = week_matches[0]; month_num = _month_number(wk_month)
            if month_num:
                temp_df = df_copy.copy()
                if 'week_of_month' not in temp_df.columns: temp_df['week_of_month'] = temp_df['date'].apply(lambda d: (d.day - 1) // 7 + 1)
                filtered_df = temp_df[(temp_df['week_of_month'] == int(wk_num)) & (temp_df['date'].dt.month == month_num)]
        if not date_filter_applied and not week_matches:
            months = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}
            filtered_dfs = [df_copy[df_copy['date'].dt.month == num].copy() for m, num in months.items() if m in ql]
            if filtered_dfs: filtered_df = pd.concat(filtered_dfs)
        for year in range(2020, 2035):
            if str(year) in query: filtered_df = filtered_df[filtered_df['date'].dt.year == year]
        if filtered_df.empty: return f"❌ No data available for your query."
        if 'week_of_month' not in filtered_df.columns: filtered_df['week_of_month'] = filtered_df['date'].apply(lambda d: (d.day - 1) // 7 + 1)

        summary_lines = []
        summary_dicts = []
        for col in matched_cols:
            s = _slice_summary(filtered_df, col)
            summary_dicts.append(s)
            if len(filtered_df) == 1:
                summary_lines.append(f"For {col.replace('_', ' ').title()} on {s['busiest_day_date']}: Total trips = {s['total']:,}.")
            else:
                summary_lines.append(
                    f"For {col.replace('_', ' ').title()}: "
                    f"Total trips = {s['total']:,} and the daily average was {s['avg']:,.2f}. "
                    f"The busiest day was {s['busiest_day_date']} ({int(s['busiest_day_value']):,} trips), "
                    f"and the quietest was {s['quietest_day_date']} ({int(s['quietest_day_value']):,} trips). "
                    f"The busiest week was Week {s['busiest_week']} ({int(s['busiest_week_trips']):,} trips)."
                )

        if "compare" in ql and len(matched_cols) > 1:
            import matplotlib.pyplot as plt
            import textwrap
            
            plot_df = pd.DataFrame({
                'Transport Mode': [m.replace('_', ' ').title() for m in matched_cols],
                'Total Trips': [s['total'] for s in summary_dicts],
                'Average Daily Trips': [s['avg'] for s in summary_dicts]
            })

            if not plot_df.empty:
                plt.style.use('seaborn-v0_8-darkgrid')
                fig, ax = plt.subplots(figsize=(12, 7))
                
                labels = [textwrap.fill(label, 15) for label in plot_df['Transport Mode']]
                metric_key, ylabel = ('Total Trips', 'Total Trips')
                if any(k in ql for k in ['average', 'avg', 'mean']):
                    metric_key, ylabel = ('Average Daily Trips', 'Average Daily Trips')

                values = plot_df[metric_key]
                colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#6366F1', '#D946EF']
                bars = ax.bar(labels, values, color=colors[:len(labels)])

                ax.set_ylabel(ylabel, fontsize=14)
                ax.set_title(f"Ridership Comparison", fontsize=18, fontweight='bold')
                
                ymax = values.max() if not values.empty else 0
                for bar, val in zip(bars, values):
                    disp = f"{int(val):,}" if metric_key != 'Average Daily Trips' else f"{val:,.2f}"
                    ax.text(bar.get_x() + bar.get_width() / 2, val + (ymax * 0.015), disp,
                            ha="center", va="bottom", fontsize=11, fontweight="bold")
                
                plt.tight_layout()
                plt.show()

        context = "\n".join(summary_lines)
        prompt = f"You are a ridership analysis assistant. Based on the following numeric context, generate a full, natural-language answer to the user query.\nCONTEXT:\n{context}\n\nUSER QUERY:\n{query}\n\nANSWER:"
        response = synthesis_model.generate_content(prompt); return response.text

    # --- ROUTE 4: Overall Ranking Questions ---
    elif "overall" in ql and "most trips" in ql:
        print("📊 Performing overall ranking...")
        modes_to_rank = [col for col in original_transport_cols]
        if not modes_to_rank: return "❌ No transport modes available to rank."

        totals = {col: df_copy[col].sum() for col in modes_to_rank}

        context_lines = [f"Total trips for {col.replace('_', ' ').title()}: {total: ,}" for col, total in totals.items()]
        context = "\n".join(context_lines)

        prompt = f"""
        You are a ridership analysis assistant. Based on the following numeric context, answer the user's question about which mode had the most trips overall by identifying the mode with the highest number.

        CONTEXT:
        {context}

        USER QUERY:
        {query}

        ANSWER:"""
        response = synthesis_model.generate_content(prompt)
        return response.text

    # --- ROUTE 5: Descriptive Queries ---
    else:
        docs = vector_retrieval(query, top_k=15)
        if not docs: return "❌ No relevant information found."
        context = "\n".join(docs)
        prompt = f"You are a ridership analysis assistant. Use ONLY the context below to answer the question in a detailed, natural-language way.\nCONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:"
        response = synthesis_model.generate_content(prompt); 
        return response.text
@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    response_text = answer_query(user_query)
    return jsonify({"answer": response_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))






