streamlit_code = '''
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import difflib
from scipy.stats import zscore

st.set_page_config(layout="wide")
st.title("📊 Shopify Plan-Based Business Insights")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Data loaded: {df.shape[0]} rows")

    def clean_currency(value):
        if isinstance(value, str):
            try:
                return float(re.sub(r'[^\d.]', '', value))
            except:
                return None
        elif isinstance(value, (float, int)):
            return float(value)
        return None

    def parse_category_hierarchy(raw_category):
        if pd.isna(raw_category) or raw_category.strip() == "":
            return {}
        first_path = raw_category.split(":")[0]
        segments = [seg.strip() for seg in first_path.split("/") if seg.strip()]
        result = {}
        if segments:
            result["Main_Category"] = segments[0]
            for i, seg in enumerate(segments[1:], start=1):
                result[f"Subcategory_{i}"] = seg
        return result

    parsed_categories = df['categories'].apply(parse_category_hierarchy)
    hierarchy_df = pd.DataFrame(parsed_categories.tolist())

    def normalize_category(name, standard_categories, threshold=0.6):
        if not name or pd.isna(name):
            return None
        match = difflib.get_close_matches(name, standard_categories, n=1, cutoff=threshold)
        return match[0] if match else name

    top_main_categories = hierarchy_df['Main_Category'].value_counts().head(10).index.tolist()
    hierarchy_df['ML_Like_Category'] = hierarchy_df['Main_Category'].apply(
        lambda x: normalize_category(x, top_main_categories)
    )

    df = pd.concat([df, hierarchy_df], axis=1)

    # --- Clean numeric columns ---
    df['monthly_app_spend_clean'] = df['monthly_app_spend'].apply(clean_currency)
    df['estimated_monthly_sales_clean'] = df['estimated_monthly_sales'].apply(clean_currency)
    df['estimated_yearly_sales_clean'] = df['estimated_yearly_sales'].apply(clean_currency)

    # --- Z-score Outlier Treatment Function ---
    def handle_outliers_zscore(df, column, threshold=3, removal_threshold=0.10):
        series = df[column].dropna()
        z_scores = zscore(series)
        mask = abs(z_scores) > threshold
        outlier_count = mask.sum()
        outlier_ratio = outlier_count / len(series)

        if outlier_ratio < removal_threshold:
            df = df.loc[~df[column].isin(series[mask])]
            # st.info(f"Removed {outlier_count} outliers from '{column}' using Z-score ({outlier_ratio:.2%})")
        else:
            mean = series.mean()
            std = series.std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            # st.info(f"Winsorized '{column}' using Z-score ({outlier_ratio:.2%} were outliers)")

        return df

    for col in ['monthly_app_spend_clean', 'estimated_monthly_sales_clean', 'estimated_yearly_sales_clean']:
        df = handle_outliers_zscore(df, col)

    analysis_type = st.radio(
        "What would you like to analyze?",
        ("Comparison", "Shopify Plus Users", "Non-Shopify Plus Users")
    )

    if analysis_type == "Shopify Plus Users":
        df_plus = df[df['plan'] == 'Shopify Plus'].copy()
        st.subheader("📈 Average Spend and Sales by Merchant")
        avg_spend = df_plus['monthly_app_spend_clean'].mean()
        avg_sales = df_plus['estimated_monthly_sales_clean'].mean()
        net = avg_sales - avg_spend

        st.dataframe(pd.DataFrame({
            "Avg Monthly App Spend": [avg_spend],
            "Avg Estimated Monthly Sales": [avg_sales],
            "Avg Net Revenue": [net]
        }))

        top = df_plus[['merchant_name', 'monthly_app_spend_clean', 'estimated_monthly_sales_clean']].dropna().head(50)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(top['merchant_name'], top['monthly_app_spend_clean'], color='orange')
            ax.set_xticklabels(top['merchant_name'], rotation=90)
            ax.set_title("Top 50 Merchants - Monthly App Spend")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(top['merchant_name'], top['estimated_monthly_sales_clean'], color='green')
            ax.set_xticklabels(top['merchant_name'], rotation=90)
            ax.set_title("Top 50 Merchants - Estimated Monthly Sales")
            st.pyplot(fig)

        st.subheader("🏙 Shopify Plus Users by City")
        st.dataframe(df_plus.groupby('city')['domain'].count().reset_index().rename(columns={'domain': 'Shopify Plus Users'}))

        st.subheader("🏙📍 City & State-wise Shopify Users by Category")
        st.dataframe(df_plus.groupby(['city', 'Main_Category']).agg(users=('domain','count'), revenue=('estimated_yearly_sales_clean','sum')).reset_index())
        st.dataframe(df_plus.groupby(['state', 'Main_Category']).agg(users=('domain','count'), revenue=('estimated_yearly_sales_clean','sum')).reset_index())

        st.subheader("📌 Monthly App Spending Vs. Avg. Product Price, No. of followers and Estimated Yearly Sales")
        c1, c2, c3 = st.columns(3)
        with c1:
            fig1, ax1 = plt.subplots()
            sns.scatterplot(data=df_plus, x='average_product_price_usd', y='monthly_app_spend_clean', ax=ax1)
            ax1.set_title("Price vs Spend")
            st.pyplot(fig1)
        with c2:
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=df_plus, x='combined_followers', y='monthly_app_spend_clean', ax=ax2)
            ax2.set_title("Followers vs Spend")
            st.pyplot(fig2)
        with c3:
            fig3, ax3 = plt.subplots()
            sns.scatterplot(data=df_plus, x='estimated_yearly_sales_clean', y='monthly_app_spend_clean', ax=ax3)
            ax3.set_title("Sales vs Spend")
            st.pyplot(fig3)

    elif analysis_type == "Non-Shopify Plus Users":
        df_non = df[df['plan'] != 'Shopify Plus'].copy()
        st.subheader("📈 Average Spend and Sales by Merchant")
        avg_spend = df_non['monthly_app_spend_clean'].mean()
        avg_sales = df_non['estimated_monthly_sales_clean'].mean()
        net = avg_sales - avg_spend

        st.dataframe(pd.DataFrame({
            "Avg Monthly App Spend": [avg_spend],
            "Avg Estimated Monthly Sales": [avg_sales],
            "Avg Net Revenue": [net]
        }))

        top = df_non[['merchant_name', 'monthly_app_spend_clean', 'estimated_monthly_sales_clean']].dropna().head(50)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(top['merchant_name'], top['monthly_app_spend_clean'], color='blue')
            ax.set_xticklabels(top['merchant_name'], rotation=90)
            ax.set_title("Top 50 Merchants - Monthly App Spend")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(top['merchant_name'], top['estimated_monthly_sales_clean'], color='purple')
            ax.set_xticklabels(top['merchant_name'], rotation=90)
            ax.set_title("Top 50 Merchants - Estimated Monthly Sales")
            st.pyplot(fig)

        st.subheader("📌 Monthly App Spending Vs. Products Sold, No. of Followers and Estimated Yearly Sales")
        c1, c2, c3 = st.columns(3)
        with c1:
            fig1, ax1 = plt.subplots()
            sns.scatterplot(data=df_non, x='products_sold', y='monthly_app_spend_clean', ax=ax1)
            ax1.set_title("Products Sold vs Spend")
            st.pyplot(fig1)
        with c2:
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=df_non, x='combined_followers', y='monthly_app_spend_clean', ax=ax2)
            ax2.set_title("Followers vs Spend")
            st.pyplot(fig2)
        with c3:
            fig3, ax3 = plt.subplots()
            sns.scatterplot(data=df_non, x='estimated_yearly_sales_clean', y='monthly_app_spend_clean', ax=ax3)
            ax3.set_title("Sales vs Spend")
            st.pyplot(fig3)

        st.subheader("🏙📍 City & State-wise Non-Shopify Users by Category")
        st.dataframe(df_non.groupby(['city', 'Main_Category']).agg(users=('domain','count'), revenue=('estimated_yearly_sales_clean','sum')).reset_index())
        st.dataframe(df_non.groupby(['state', 'Main_Category']).agg(users=('domain','count'), revenue=('estimated_yearly_sales_clean','sum')).reset_index())

    elif analysis_type == "Comparison":
        df_plus = df[df['plan'] == 'Shopify Plus'].copy()
        df_non = df[df['plan'] != 'Shopify Plus'].copy()

        st.subheader("📉 Average Spend and Sales Comparison")
        def avg_stats(data, label):
            return {
                'Plan': label,
                'Avg Spend': data['monthly_app_spend_clean'].mean(),
                'Avg Sales': data['estimated_monthly_sales_clean'].mean(),
                'Avg Net': data['estimated_monthly_sales_clean'].mean() - data['monthly_app_spend_clean'].mean()
            }
        compare_df = pd.DataFrame([avg_stats(df_plus, 'Shopify Plus'), avg_stats(df_non, 'Non-Shopify')])
        st.dataframe(compare_df)
        st.markdown("## 🧾 Overall Business Insights Across All Categories")

        # Bar chart – State vs Yearly Sales
        st.subheader("📍 Total Estimated Yearly Sales by State")
        state_sales = df.groupby("state")["estimated_yearly_sales_clean"].sum().sort_values(ascending=False).reset_index()
        fig1, ax1 = plt.subplots(figsize=(18,12))
        sns.barplot(data=state_sales, x="state", y="estimated_yearly_sales_clean", palette="Blues_r", ax=ax1)
        ax1.set_title("State-wise Yearly Sales")
        ax1.set_ylabel("Total Yearly Sales")
        ax1.set_xlabel("State")
        plt.xticks(rotation=45)
        st.pyplot(fig1)
    
        # Bar chart – Main Category vs Yearly Sales
        st.subheader("📦 Total Estimated Yearly Sales by Main Category")
        cat_sales = df.groupby("Main_Category")["estimated_yearly_sales_clean"].sum().sort_values(ascending=False).reset_index()
        fig2, ax2 = plt.subplots(figsize=(18,12))
        sns.barplot(data=cat_sales, x="Main_Category", y="estimated_yearly_sales_clean", palette="Greens", ax=ax2)
        ax2.set_title("Main Category-wise Yearly Sales")
        ax2.set_ylabel("Total Yearly Sales")
        ax2.set_xlabel("Main Category")
        plt.xticks(rotation=45)
        st.pyplot(fig2)


        st.subheader("📊 Additional Metrics Breakdown")
        total_users = len(df)
        users_with_plan = len(df_plus)
        percent_with_plan = (users_with_plan / total_users) * 100
        total_revenue_with_plan = df_plus['estimated_yearly_sales_clean'].sum()
        avg_rev_per_user = {
            'Shopify Plus': df_plus['estimated_yearly_sales_clean'].mean(),
            'Non-Shopify': df_non['estimated_yearly_sales_clean'].mean()
        }
        avg_followers = {
            'Shopify Plus': df_plus['combined_followers'].mean(),
            'Non-Shopify': df_non['combined_followers'].mean()
        }
        metric_df = pd.DataFrame({
            'Metric': [
                '% of Users with Plan',
                'Total Revenue from Plans',
                'Avg Revenue Per User (Shopify Plus)',
                'Avg Revenue Per User (Non-Shopify)',
                'Avg Followers (Shopify Plus)',
                'Avg Followers (Non-Shopify)'
            ],
            'Value': [
                f"{percent_with_plan:.2f}%",
                f"${total_revenue_with_plan:,.2f}",
                f"${avg_rev_per_user['Shopify Plus']:,.2f}",
                f"${avg_rev_per_user['Non-Shopify']:,.2f}",
                f"{avg_followers['Shopify Plus']:,.2f}",
                f"{avg_followers['Non-Shopify']:,.2f}"
            ]
        })
        st.dataframe(metric_df)

        st.subheader("📈 ROI Analysis by User")
        # st.subheader("ROI=((estimated yearly sales-[Monthly App Spend]*12)/Monthly App Spend*12)")
        df['ROI'] = (df['estimated_yearly_sales_clean'] - df['monthly_app_spend_clean'] * 12) / (df['monthly_app_spend_clean'] * 12)
        df_roi = df[~df['ROI'].isna() & (df['monthly_app_spend_clean'] > 0)]

        st.subheader("📈 ROI Distribution (Box Plot)")
        fig, ax = plt.subplots(figsize=(18,12))
        sns.boxplot(x=df_roi['ROI'], color='lightcoral', ax=ax)
        ax.set_title("ROI Distribution (Box Plot)")
        ax.set_xlabel("ROI")
        st.pyplot(fig)

        st.subheader("📌 Average ROI by Main Category")
        if 'Main_Category' in df_roi.columns:
            roi_category = df_roi.groupby('Main_Category')['ROI'].mean().reset_index().sort_values(by='ROI', ascending=False)
            st.dataframe(roi_category)

        st.subheader("📍 Average ROI by State")
        roi_state = df_roi.groupby('state')['ROI'].mean().reset_index().sort_values(by='ROI', ascending=False)
        st.dataframe(roi_state)

    st.markdown("---")
    st.markdown("Built by Medhavi")
else:
    st.info("Please upload a CSV file to begin.")

'''
# ✅ Write to app.py
with open("app.py", "w", encoding="utf-8") as f:
    f.write(streamlit_code)

print("✅ app.py file has been created with extended Shopify Plus analysis.").
