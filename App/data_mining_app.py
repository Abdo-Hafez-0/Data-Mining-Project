import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Question Difficulty Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Question Difficulty Prediction Pipeline")
st.markdown("---")

# Initialize session state
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'featured_df' not in st.session_state:
    st.session_state.featured_df = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# Sidebar Navigation
st.sidebar.title("🔧 Pipeline Steps")
step = st.sidebar.radio(
    "Select Step:",
    ["1. Upload Data", "2. Data Cleaning", "3. EDA", "4. Feature Engineering", "5. Model Training"]
)

# =============================================================================
# STEP 1: UPLOAD DATA
# =============================================================================
if step == "1. Upload Data":
    st.header("📁 Step 1: Upload Your Data")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        st.session_state.raw_df = pd.read_csv(uploaded_file)
        st.success(f"✅ Data loaded! Shape: {st.session_state.raw_df.shape}")
        
        st.subheader("Data Preview")
        st.dataframe(st.session_state.raw_df.head(10))
        
        st.subheader("Data Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", st.session_state.raw_df.shape[0])
        with col2:
            st.metric("Columns", st.session_state.raw_df.shape[1])
        with col3:
            st.metric("Missing Values", st.session_state.raw_df.isnull().sum().sum())
        
        st.subheader("Column Types")
        dtype_df = pd.DataFrame({
            'Column': st.session_state.raw_df.columns,
            'Type': st.session_state.raw_df.dtypes.values,
            'Missing': st.session_state.raw_df.isnull().sum().values,
            'Unique': st.session_state.raw_df.nunique().values
        })
        st.dataframe(dtype_df)

# =============================================================================
# STEP 2: DATA CLEANING
# =============================================================================
elif step == "2. Data Cleaning":
    st.header("🧹 Step 2: Data Cleaning")
    
    if st.session_state.raw_df is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.raw_df.copy()
        
        if st.session_state.cleaned_df is not None:
            df = st.session_state.cleaned_df.copy()
        
        st.subheader("🎛️ Cleaning Controls")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Missing Values", "Outliers", "Duplicates", "Transformations", "Filters"
        ])
        
        # TAB 1: Missing Values
        with tab1:
            st.markdown("### Handle Missing Values")
            
            missing_df = df.isnull().sum()
            missing_df = missing_df[missing_df > 0]
            
            if len(missing_df) > 0:
                st.write("Columns with missing values:")
                st.dataframe(pd.DataFrame({
                    'Column': missing_df.index,
                    'Missing Count': missing_df.values,
                    'Missing %': (missing_df.values / len(df) * 100).round(2)
                }))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    missing_scope = st.radio("Apply to:", ["Entire Dataset", "Specific Column"])
                
                with col2:
                    missing_strategy = st.selectbox(
                        "Strategy:",
                        ["median", "mean", "mode", "drop", "ffill", "bfill", "constant"]
                    )
                
                if missing_scope == "Specific Column":
                    selected_col = st.selectbox("Select Column:", missing_df.index.tolist())
                    columns_to_clean = [selected_col]
                else:
                    columns_to_clean = missing_df.index.tolist()
                
                fill_value = 0
                if missing_strategy == "constant":
                    fill_value = st.number_input("Fill Value:", value=0)
                
                if st.button("Apply Missing Value Treatment", key="missing_btn"):
                    for col in columns_to_clean:
                        if missing_strategy == "median":
                            if np.issubdtype(df[col].dtype, np.number):
                                df[col] = df[col].fillna(df[col].median())
                        elif missing_strategy == "mean":
                            if np.issubdtype(df[col].dtype, np.number):
                                df[col] = df[col].fillna(df[col].mean())
                        elif missing_strategy == "mode":
                            if len(df[col].mode()) > 0:
                                df[col] = df[col].fillna(df[col].mode().iloc[0])
                        elif missing_strategy == "drop":
                            df = df.dropna(subset=[col])
                        elif missing_strategy == "ffill":
                            df[col] = df[col].ffill()
                        elif missing_strategy == "bfill":
                            df[col] = df[col].bfill()
                        elif missing_strategy == "constant":
                            df[col] = df[col].fillna(fill_value)
                    
                    st.session_state.cleaned_df = df
                    st.success("✅ Missing values handled!")
                    st.rerun()
            else:
                st.info("No missing values found!")
        
        # TAB 2: Outliers
        with tab2:
            st.markdown("### Handle Outliers")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    outlier_col = st.selectbox("Select Column:", numeric_cols, key="outlier_col")
                
                with col2:
                    outlier_method = st.selectbox("Detection Method:", ["IQR", "Z-Score", "Percentile"])
                
                col3, col4 = st.columns(2)
                
                with col3:
                    if outlier_method == "IQR":
                        threshold = st.slider("IQR Multiplier:", 1.0, 3.0, 1.5, 0.1)
                    elif outlier_method == "Z-Score":
                        threshold = st.slider("Z-Score Threshold:", 1.0, 5.0, 3.0, 0.1)
                    else:
                        threshold = st.slider("Percentile:", 0.90, 0.99, 0.99, 0.01)
                
                with col4:
                    outlier_action = st.selectbox(
                        "Action:",
                        ["Remove", "Cap", "Replace with Median", "Replace with Mean"]
                    )
                
                data = df[outlier_col].dropna()
                
                if len(data) > 0:
                    if outlier_method == "IQR":
                        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower, upper = Q1 - threshold * IQR, Q3 + threshold * IQR
                    elif outlier_method == "Z-Score":
                        mean, std = data.mean(), data.std()
                        if std > 0:
                            lower, upper = mean - threshold * std, mean + threshold * std
                        else:
                            lower, upper = data.min(), data.max()
                    else:
                        lower = data.quantile(1 - threshold)
                        upper = data.quantile(threshold)
                    
                    outliers = data[(data < lower) | (data > upper)]
                    
                    st.write(f"**Outlier Statistics for {outlier_col}:**")
                    st.write(f"- Lower Bound: {lower:.2f}")
                    st.write(f"- Upper Bound: {upper:.2f}")
                    st.write(f"- Outliers Found: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)")
                    
                    if st.button("Apply Outlier Treatment", key="outlier_btn"):
                        if outlier_action == "Remove":
                            df = df[(df[outlier_col] >= lower) & (df[outlier_col] <= upper)]
                        elif outlier_action == "Cap":
                            df[outlier_col] = df[outlier_col].clip(lower, upper)
                        elif outlier_action == "Replace with Median":
                            mask = (df[outlier_col] < lower) | (df[outlier_col] > upper)
                            df.loc[mask, outlier_col] = df[outlier_col].median()
                        elif outlier_action == "Replace with Mean":
                            mask = (df[outlier_col] < lower) | (df[outlier_col] > upper)
                            df.loc[mask, outlier_col] = df[outlier_col].mean()
                        
                        st.session_state.cleaned_df = df
                        st.success("✅ Outliers handled!")
                        st.rerun()
            else:
                st.info("No numeric columns found.")
        
        # TAB 3: Duplicates
        with tab3:
            st.markdown("### Remove Duplicates")
            
            dup_count = df.duplicated().sum()
            st.write(f"**Total Duplicate Rows:** {dup_count}")
            
            default_cols = []
            if all(c in df.columns for c in ['user_id', 'question_id', 'timestamp']):
                default_cols = ['user_id', 'question_id', 'timestamp']
            
            col1, col2 = st.columns(2)
            
            with col1:
                dup_subset = st.multiselect(
                    "Check duplicates based on columns:",
                    df.columns.tolist(),
                    default=default_cols
                )
            
            with col2:
                keep_option = st.selectbox("Keep:", ["first", "last", "none (remove all)"])
            
            if dup_subset:
                subset_dups = df.duplicated(subset=dup_subset).sum()
                st.write(f"**Duplicates based on selected columns:** {subset_dups}")
            
            if st.button("Remove Duplicates", key="dup_btn"):
                keep_val = keep_option if keep_option != "none (remove all)" else False
                df = df.drop_duplicates(subset=dup_subset if dup_subset else None, keep=keep_val)
                st.session_state.cleaned_df = df
                st.success(f"✅ Duplicates removed! New shape: {df.shape}")
                st.rerun()
        
        # TAB 4: Transformations
        with tab4:
            st.markdown("### Data Transformations")
            
            st.markdown("#### Convert Timestamp")
            potential_ts_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            
            if potential_ts_cols:
                ts_col = st.selectbox("Select timestamp column:", potential_ts_cols)
                ts_unit = st.selectbox("Timestamp unit:", ["ms", "s", "us", "ns"])
                
                if st.button("Convert Timestamp", key="ts_btn"):
                    try:
                        df[ts_col] = pd.to_datetime(df[ts_col], unit=ts_unit)
                        st.session_state.cleaned_df = df
                        st.success("✅ Timestamp converted!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.info("No timestamp columns detected.")
            
            st.markdown("#### Create Elapsed Time in Seconds")
            
            if 'elapsed_time' in df.columns:
                divisor = st.number_input("Divisor (e.g., 1000 for ms to s):", value=1000)
                
                if st.button("Create elapsed_time_seconds", key="elapsed_btn"):
                    df['elapsed_time_seconds'] = df['elapsed_time'] / divisor
                    df = df.drop(columns=['elapsed_time'])
                    st.session_state.cleaned_df = df
                    st.success("✅ elapsed_time_seconds created!")
                    st.rerun()
            else:
                st.info("No 'elapsed_time' column found.")
        
        # TAB 5: Filters
        with tab5:
            st.markdown("### Filter Data")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                filter_col = st.selectbox("Select column to filter:", numeric_cols)
                
                col_min = float(df[filter_col].min()) if pd.notna(df[filter_col].min()) else 0.0
                col_max = float(df[filter_col].max()) if pd.notna(df[filter_col].max()) else 1.0
                
                col1, col2 = st.columns(2)
                
                with col1:
                    min_val = st.number_input("Minimum value:", value=col_min, key="filter_min")
                with col2:
                    max_val = st.number_input("Maximum value:", value=col_max, key="filter_max")
                
                if st.button("Apply Filter", key="filter_btn"):
                    df = df[(df[filter_col] >= min_val) & (df[filter_col] <= max_val)]
                    st.session_state.cleaned_df = df
                    st.success(f"✅ Filter applied! New shape: {df.shape}")
                    st.rerun()
            else:
                st.info("No numeric columns found.")
        
        # Show current data state
        st.markdown("---")
        st.subheader("📋 Current Data State")
        
        current_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else df
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", current_df.shape[0])
        with col2:
            st.metric("Columns", current_df.shape[1])
        with col3:
            st.metric("Missing", current_df.isnull().sum().sum())
        
        st.dataframe(current_df.head(10))
        
        if st.button("💾 Finalize Cleaning", key="finalize_clean"):
            st.session_state.cleaned_df = current_df
            st.success("✅ Cleaned data saved!")

# =============================================================================
# STEP 3: EDA
# =============================================================================
elif step == "3. EDA":
    st.header("📈 Step 3: Exploratory Data Analysis")
    
    if st.session_state.cleaned_df is None and st.session_state.raw_df is None:
        st.warning("⚠️ Please upload and clean data first!")
    else:
        df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df
        df = df.copy()
        
        tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlation", "Categorical", "Custom Plots"])
        
        # TAB 1: Distribution
        with tab1:
            st.subheader("Distribution Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    dist_col = st.selectbox("Select Column:", numeric_cols, key="dist_col")
                with col2:
                    plot_type = st.selectbox("Plot Type:", ["Histogram", "Box Plot", "Violin Plot", "KDE"])
                
                col3, col4 = st.columns(2)
                
                with col3:
                    bins = st.slider("Number of Bins:", 10, 100, 30)
                with col4:
                    color = st.color_picker("Color:", "#636EFA")
                
                try:
                    if plot_type == "Histogram":
                        fig = px.histogram(df, x=dist_col, nbins=bins, title=f'Distribution of {dist_col}', color_discrete_sequence=[color])
                    elif plot_type == "Box Plot":
                        fig = px.box(df, y=dist_col, title=f'Box Plot of {dist_col}', color_discrete_sequence=[color])
                    elif plot_type == "Violin Plot":
                        fig = px.violin(df, y=dist_col, title=f'Violin Plot of {dist_col}', color_discrete_sequence=[color], box=True)
                    else:
                        fig = px.histogram(df, x=dist_col, nbins=bins, title=f'KDE of {dist_col}', histnorm='probability density', color_discrete_sequence=[color])
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                
                st.subheader(f"Statistics for {dist_col}")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Mean", f"{df[dist_col].mean():.2f}")
                with c2:
                    st.metric("Median", f"{df[dist_col].median():.2f}")
                with c3:
                    st.metric("Std Dev", f"{df[dist_col].std():.2f}")
                with c4:
                    st.metric("Skewness", f"{df[dist_col].skew():.2f}")
            else:
                st.info("No numeric columns found.")
        
        # TAB 2: Correlation
        with tab2:
            st.subheader("Correlation Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    corr_method = st.selectbox("Correlation Method:", ["pearson", "spearman", "kendall"])
                with col2:
                    color_scale = st.selectbox("Color Scale:", ["RdBu_r", "Viridis", "Blues", "Reds"])
                
                selected_cols = st.multiselect(
                    "Select columns:",
                    numeric_cols,
                    default=numeric_cols[:min(6, len(numeric_cols))]
                )
                
                if len(selected_cols) >= 2:
                    try:
                        corr_matrix = df[selected_cols].corr(method=corr_method)
                        fig = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale=color_scale, title=f'Correlation Matrix ({corr_method})')
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                    
                    st.subheader("Scatter Plot")
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        x_col = st.selectbox("X-axis:", selected_cols, key="scatter_x")
                    with sc2:
                        y_options = [c for c in selected_cols if c != x_col]
                        y_col = st.selectbox("Y-axis:", y_options if y_options else selected_cols, key="scatter_y")
                    
                    trendline = st.checkbox("Add Trendline")
                    
                    try:
                        fig = px.scatter(df, x=x_col, y=y_col, trendline="ols" if trendline else None, title=f'{y_col} vs {x_col}')
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.info("Need at least 2 numeric columns.")
        
        # TAB 3: Categorical
        with tab3:
            st.subheader("Categorical Analysis")
            
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].nunique() <= 20:
                    cat_cols.append(col)
            
            if cat_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    cat_col = st.selectbox("Select Column:", cat_cols, key="cat_col")
                with col2:
                    cat_plot_type = st.selectbox("Plot Type:", ["Bar Chart", "Pie Chart", "Treemap"])
                
                top_n = st.slider("Top N categories:", 5, 30, 10)
                
                try:
                    value_counts = df[cat_col].value_counts().head(top_n)
                    
                    if cat_plot_type == "Bar Chart":
                        fig = px.bar(x=value_counts.index.astype(str), y=value_counts.values, title=f'Distribution of {cat_col}', labels={'x': cat_col, 'y': 'Count'})
                    elif cat_plot_type == "Pie Chart":
                        fig = px.pie(values=value_counts.values, names=value_counts.index.astype(str), title=f'Distribution of {cat_col}')
                    else:
                        fig = px.treemap(names=value_counts.index.astype(str), parents=[''] * len(value_counts), values=value_counts.values, title=f'Treemap of {cat_col}')
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.info("No categorical columns found.")
        
        # TAB 4: Custom Plots (FIXED)
        with tab4:
            st.subheader("Custom Visualizations")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df.columns.tolist()
            
            custom_plot_type = st.selectbox("Plot Type:", ["Grouped Bar", "Box by Category", "Pair Plot", "Line Plot"])
            
            if custom_plot_type == "Grouped Bar":
                col1, col2 = st.columns(2)
                
                with col1:
                    group_col = st.selectbox("Group by:", all_cols, key="group_col")
                with col2:
                    value_options = [c for c in numeric_cols if c != group_col]
                    if not value_options:
                        value_options = numeric_cols
                    value_col = st.selectbox("Value:", value_options, key="value_col") if value_options else None
                
                agg_func = st.selectbox("Aggregation:", ["mean", "sum", "count", "median"])
                
                if value_col:
                    try:
                        # FIXED: Avoid reset_index conflict
                        agg_result = df.groupby(group_col)[value_col].agg(agg_func)
                        grouped = pd.DataFrame({
                            group_col: agg_result.index.astype(str),
                            value_col: agg_result.values
                        })
                        
                        fig = px.bar(grouped, x=group_col, y=value_col, title=f'{agg_func.capitalize()} of {value_col} by {group_col}')
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            elif custom_plot_type == "Box by Category":
                col1, col2 = st.columns(2)
                
                with col1:
                    cat_col = st.selectbox("Category:", all_cols, key="box_cat")
                with col2:
                    num_options = [c for c in numeric_cols if c != cat_col]
                    num_col = st.selectbox("Numeric:", num_options if num_options else numeric_cols, key="box_num") if numeric_cols else None
                
                if num_col:
                    try:
                        fig = px.box(df, x=cat_col, y=num_col, title=f'{num_col} by {cat_col}')
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            elif custom_plot_type == "Pair Plot":
                selected = st.multiselect("Select columns (2-5):", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))] if numeric_cols else [])
                
                if len(selected) >= 2:
                    try:
                        fig = px.scatter_matrix(df[selected], title="Pair Plot")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            elif custom_plot_type == "Line Plot":
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("X-axis:", all_cols, key="line_x")
                with col2:
                    y_options = [c for c in numeric_cols if c != x_col]
                    y_col = st.selectbox("Y-axis:", y_options if y_options else numeric_cols, key="line_y") if numeric_cols else None
                
                if y_col:
                    try:
                        fig = px.line(df.sort_values(x_col), x=x_col, y=y_col, title=f'{y_col} over {x_col}')
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                # =============================================================================
# STEP 4: FEATURE ENGINEERING
# =============================================================================
elif step == "4. Feature Engineering":
    st.header("⚙️ Step 4: Feature Engineering")
    
    if st.session_state.cleaned_df is None and st.session_state.raw_df is None:
        st.warning("⚠️ Please upload and clean data first!")
    else:
        df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df
        df = df.copy()
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Infer Correct Answers",
            "Aggregate Features", 
            "Engineer Features",
            "Create Target"
        ])
        
        # TAB 1: Infer Correct Answers
        with tab1:
            st.subheader("Infer Correct Answers (Majority Voting)")
            
            required_cols = ['question_id', 'user_answer']
            
            if all(col in df.columns for col in required_cols):
                min_attempts = st.slider("Minimum attempts per question:", 1, 50, 10)
                
                if st.button("Infer Correct Answers", key="infer_btn"):
                    try:
                        # Infer correct answers
                        answer_counts = df.groupby(['question_id', 'user_answer']).size()
                        
                        correct_answers = (
                            answer_counts
                            .groupby(level=0)
                            .idxmax()
                            .apply(lambda x: x[1])
                            .reset_index(name='assumed_correct_answer')
                        )
                        
                        # Filter by minimum attempts
                        attempts_per_question = df['question_id'].value_counts()
                        valid_questions = attempts_per_question[
                            attempts_per_question >= min_attempts
                        ].index
                        
                        df = df[df['question_id'].isin(valid_questions)]
                        
                        # Merge and create is_correct
                        df = df.merge(correct_answers, on='question_id', how='left')
                        df['is_correct'] = (
                            df['user_answer'] == df['assumed_correct_answer']
                        ).astype(int)
                        
                        st.session_state.cleaned_df = df
                        
                        st.success(f"✅ Correct answers inferred!")
                        st.write(f"- Valid questions: {len(valid_questions)}")
                        st.write(f"- Overall accuracy: {df['is_correct'].mean():.2%}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error(f"Required columns not found: {required_cols}")
                st.write(f"Available columns: {df.columns.tolist()}")
        
        # TAB 2: Aggregate Features
        with tab2:
            st.subheader("Aggregate to Question Level")
            
            if 'is_correct' not in df.columns:
                st.warning("⚠️ Please infer correct answers first!")
            elif 'elapsed_time_seconds' not in df.columns:
                st.warning("⚠️ elapsed_time_seconds column not found!")
            else:
                st.write("Select aggregation functions:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    agg_count = st.checkbox("Count (attempts)", value=True)
                    agg_mean = st.checkbox("Mean time", value=True)
                    agg_median = st.checkbox("Median time", value=True)
                
                with col2:
                    agg_std = st.checkbox("Std time", value=True)
                    agg_min = st.checkbox("Min time", value=True)
                    agg_max = st.checkbox("Max time", value=True)
                
                if st.button("Aggregate Data", key="agg_btn"):
                    try:
                        agg_dict = {'is_correct': 'mean'}
                        
                        if agg_count:
                            agg_dict['user_id'] = 'count'
                        
                        time_aggs = []
                        if agg_mean:
                            time_aggs.append('mean')
                        if agg_median:
                            time_aggs.append('median')
                        if agg_std:
                            time_aggs.append('std')
                        if agg_min:
                            time_aggs.append('min')
                        if agg_max:
                            time_aggs.append('max')
                        
                        if time_aggs:
                            agg_dict['elapsed_time_seconds'] = time_aggs
                        
                        question_df = df.groupby('question_id').agg(agg_dict)
                        
                        # Flatten columns
                        new_columns = []
                        for col in question_df.columns:
                            if isinstance(col, tuple):
                                new_columns.append(f"{col[0]}_{col[1]}")
                            else:
                                new_columns.append(col)
                        question_df.columns = new_columns
                        
                        question_df = question_df.reset_index()
                        
                        # Rename columns
                        rename_dict = {
                            'user_id_count': 'attempts',
                            'is_correct_mean': 'success_rate',
                            'elapsed_time_seconds_mean': 'avg_time',
                            'elapsed_time_seconds_median': 'median_time',
                            'elapsed_time_seconds_std': 'time_std',
                            'elapsed_time_seconds_min': 'min_time',
                            'elapsed_time_seconds_max': 'max_time'
                        }
                        
                        question_df = question_df.rename(columns={
                            k: v for k, v in rename_dict.items() if k in question_df.columns
                        })
                        
                        st.session_state.featured_df = question_df
                        
                        st.success(f"✅ Aggregated! Shape: {question_df.shape}")
                        st.dataframe(question_df.head(10))
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # TAB 3: Engineer Features
        with tab3:
            st.subheader("Create Engineered Features")
            
            if st.session_state.featured_df is None:
                st.warning("⚠️ Please aggregate data first!")
            else:
                question_df = st.session_state.featured_df.copy()
                
                st.write("Select features to create:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    create_log_attempts = st.checkbox("Log Attempts", value=True)
                    create_time_skew = st.checkbox("Time Skew", value=True)
                
                with col2:
                    create_time_range = st.checkbox("Time Range", value=True)
                    create_relative_var = st.checkbox("Relative Time Variance", value=True)
                
                if st.button("Create Features", key="feat_btn"):
                    try:
                        if create_log_attempts and 'attempts' in question_df.columns:
                            question_df['log_attempts'] = np.log1p(question_df['attempts'])
                        
                        if create_time_skew and all(c in question_df.columns for c in ['avg_time', 'median_time']):
                            question_df['time_skew'] = question_df['avg_time'] - question_df['median_time']
                        
                        if create_time_range and all(c in question_df.columns for c in ['max_time', 'min_time']):
                            question_df['time_range'] = question_df['max_time'] - question_df['min_time']
                        
                        if create_relative_var and all(c in question_df.columns for c in ['time_std', 'avg_time']):
                            question_df['relative_time_variance'] = (
                                question_df['time_std'] / question_df['avg_time'].replace(0, np.nan)
                            )
                        
                        st.session_state.featured_df = question_df
                        
                        st.success("✅ Features created!")
                        st.dataframe(question_df.head(10))
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # TAB 4: Create Target
        with tab4:
            st.subheader("Create Difficulty Target")
            
            if st.session_state.featured_df is None:
                st.warning("⚠️ Please create features first!")
            else:
                question_df = st.session_state.featured_df.copy()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    target_method = st.selectbox(
                        "Target Creation Method:",
                        ["Composite (Success + Time)", "Success Rate Only", "Time Only"]
                    )
                
                with col2:
                    n_classes = st.selectbox("Number of Classes:", [2, 3, 4, 5], index=1)
                
                success_weight = 1.0
                time_weight = 1.0
                
                if target_method == "Composite (Success + Time)":
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        success_weight = st.slider("Success Rate Weight:", 0.0, 2.0, 1.0, 0.1)
                    with col4:
                        time_weight = st.slider("Time Weight:", 0.0, 2.0, 1.0, 0.1)
                
                if st.button("Create Target", key="target_btn"):
                    try:
                        scaler = StandardScaler()
                        
                        if target_method == "Composite (Success + Time)":
                            if 'success_rate' in question_df.columns and 'avg_time' in question_df.columns:
                                temp_df = question_df[['success_rate', 'avg_time']].fillna(
                                    question_df[['success_rate', 'avg_time']].median()
                                )
                                
                                scaled = scaler.fit_transform(temp_df)
                                question_df['success_z'] = scaled[:, 0]
                                question_df['time_z'] = scaled[:, 1]
                                
                                question_df['difficulty'] = (
                                    (-question_df['success_z'] * success_weight) + 
                                    (question_df['time_z'] * time_weight)
                                )
                            else:
                                st.error("Required columns not found!")
                                st.stop()
                        
                        elif target_method == "Success Rate Only":
                            if 'success_rate' in question_df.columns:
                                question_df['difficulty'] = 1 - question_df['success_rate']
                            else:
                                st.error("success_rate column not found!")
                                st.stop()
                        
                        else:  # Time Only
                            if 'avg_time' in question_df.columns:
                                temp_df = question_df[['avg_time']].fillna(question_df['avg_time'].median())
                                scaled = scaler.fit_transform(temp_df)
                                question_df['difficulty'] = scaled[:, 0]
                            else:
                                st.error("avg_time column not found!")
                                st.stop()
                        
                        # Create classes
                        labels = ['Easy', 'Medium', 'Hard', 'Very Hard', 'Expert'][:n_classes]
                        
                        question_df['difficulty_class'] = pd.qcut(
                            question_df['difficulty'],
                            q=n_classes,
                            labels=labels,
                            duplicates='drop'
                        )
                        
                        st.session_state.featured_df = question_df
                        
                        st.success("✅ Target created!")
                        
                        # Show distribution
                        fig = px.histogram(
                            question_df, 
                            x='difficulty_class',
                            title='Difficulty Class Distribution',
                            color='difficulty_class'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(question_df.head(10))
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# =============================================================================
# STEP 5: MODEL TRAINING
# =============================================================================
elif step == "5. Model Training":
    st.header("🤖 Step 5: Model Training & Results")
    
    if st.session_state.featured_df is None:
        st.warning("⚠️ Please complete feature engineering first!")
    elif 'difficulty_class' not in st.session_state.featured_df.columns:
        st.warning("⚠️ Please create difficulty target first!")
    else:
        df = st.session_state.featured_df.copy()
        
        tab1, tab2, tab3 = st.tabs(["Configure", "Train", "Results"])
        
        # TAB 1: Configure
        with tab1:
            st.subheader("Model Configuration")
            
            st.markdown("### Select Features")
            
            # Get safe features
            unsafe_cols = ['difficulty', 'difficulty_class', 'success_rate', 
                          'avg_time', 'success_z', 'time_z', 'question_id']
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            safe_features = [col for col in numeric_cols if col not in unsafe_cols]
            
            st.warning("⚠️ Avoid using features that were used to create the target (data leakage)")
            
            selected_features = st.multiselect(
                "Select features for training:",
                safe_features,
                default=safe_features[:min(5, len(safe_features))]
            )
            
            st.markdown("### Select Models")
            
            available_models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
            }
            
            selected_models = st.multiselect(
                "Select models to train:",
                list(available_models.keys()),
                default=['Random Forest', 'Logistic Regression']
            )
            
            st.markdown("### Train/Test Split")
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_size = st.slider("Test Size:", 0.1, 0.4, 0.2, 0.05)
            with col2:
                cv_folds = st.slider("Cross-Validation Folds:", 2, 10, 5)
            
            st.session_state.model_config = {
                'features': selected_features,
                'models': {k: v for k, v in available_models.items() if k in selected_models},
                'test_size': test_size,
                'cv_folds': cv_folds
            }
            
            if selected_features and selected_models:
                st.success("✅ Configuration saved!")
            else:
                st.warning("Please select at least one feature and one model.")
        
        # TAB 2: Train
        with tab2:
            st.subheader("Train Models")
            
            if 'model_config' not in st.session_state:
                st.warning("⚠️ Please configure models first!")
            elif len(st.session_state.model_config.get('features', [])) == 0:
                st.warning("⚠️ Please select at least one feature!")
            elif len(st.session_state.model_config.get('models', {})) == 0:
                st.warning("⚠️ Please select at least one model!")
            else:
                config = st.session_state.model_config
                
                st.write(f"**Features:** {config['features']}")
                st.write(f"**Models:** {list(config['models'].keys())}")
                st.write(f"**Test Size:** {config['test_size']}")
                st.write(f"**CV Folds:** {config['cv_folds']}")
                
                if st.button("🚀 Train All Models", key="train_btn"):
                    # Prepare data
                    X = df[config['features']].fillna(df[config['features']].median())
                    
                    le = LabelEncoder()
                    y = le.fit_transform(df['difficulty_class'])
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=config['test_size'],
                        stratify=y,
                        random_state=42
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    results = {}
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, (name, model) in enumerate(config['models'].items()):
                        status_text.text(f"Training {name}...")
                        
                        # Train
                        model.fit(X_train_scaled, y_train)
                        
                        # Predict
                        y_pred = model.predict(X_test_scaled)
                        
                        # Cross-validation
                        cv_scores = cross_val_score(
                            model, X_train_scaled, y_train, 
                            cv=config['cv_folds']
                        )
                        
                        # Metrics
                        results[name] = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                            'cv_mean': cv_scores.mean(),
                            'cv_std': cv_scores.std(),
                            'confusion_matrix': confusion_matrix(y_test, y_pred),
                            'model': model,
                            'classes': le.classes_
                        }
                        
                        # Feature importance
                        if hasattr(model, 'feature_importances_'):
                            results[name]['feature_importance'] = dict(
                                zip(config['features'], model.feature_importances_)
                            )
                        
                        progress_bar.progress((i + 1) / len(config['models']))
                    
                    st.session_state.model_results = results
                    status_text.text("✅ Training complete!")
                    st.success("All models trained successfully!")
        
        # TAB 3: Results
        with tab3:
            st.subheader("Model Results Comparison")
            
            if not st.session_state.model_results:
                st.warning("⚠️ Please train models first!")
            else:
                results = st.session_state.model_results
                
                # Summary table
                st.markdown("### 📊 Performance Summary")
                
                summary_data = []
                for name, metrics in results.items():
                    summary_data.append({
                        'Model': name,
                        'Accuracy': f"{metrics['accuracy']:.4f}",
                        'Precision': f"{metrics['precision']:.4f}",
                        'Recall': f"{metrics['recall']:.4f}",
                        'F1 Score': f"{metrics['f1']:.4f}",
                        'CV Mean': f"{metrics['cv_mean']:.4f}",
                        'CV Std': f"{metrics['cv_std']:.4f}"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Best model
                best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
                st.success(f"🏆 Best Model: **{best_model[0]}** with accuracy {best_model[1]['accuracy']:.4f}")
                
                # Comparison chart
                st.markdown("### 📈 Performance Comparison")
                
                metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
                
                plot_data = []
                for name, metrics in results.items():
                    for metric in metrics_to_plot:
                        plot_data.append({
                            'Model': name,
                            'Metric': metric.capitalize(),
                            'Value': metrics[metric]
                        })
                
                plot_df = pd.DataFrame(plot_data)
                
                fig = px.bar(
                    plot_df, 
                    x='Model', 
                    y='Value', 
                    color='Metric',
                    barmode='group',
                    title='Model Performance Comparison'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Individual model details
                st.markdown("### 🔍 Detailed Results")
                
                selected_model = st.selectbox(
                    "Select model for detailed view:",
                    list(results.keys())
                )
                
                if selected_model:
                    model_results = results[selected_model]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Confusion Matrix")
                        
                        cm = model_results['confusion_matrix']
                        classes = model_results['classes']
                        
                        fig_cm = px.imshow(
                            cm,
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=classes,
                            y=classes,
                            text_auto=True,
                            color_continuous_scale='Blues'
                        )
                        fig_cm.update_layout(height=400)
                        
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    with col2:
                        if 'feature_importance' in model_results:
                            st.markdown("#### Feature Importance")
                            
                            importance = model_results['feature_importance']
                            importance_df = pd.DataFrame({
                                'Feature': list(importance.keys()),
                                'Importance': list(importance.values())
                            }).sort_values('Importance', ascending=True)
                            
                            fig_imp = px.bar(
                                importance_df,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Feature Importance',
                                color='Importance',
                                color_continuous_scale='Viridis'
                            )
                            fig_imp.update_layout(height=400)
                            
                            st.plotly_chart(fig_imp, use_container_width=True)
                        else:
                            st.info("Feature importance not available for this model.")
                
                # Download results
                st.markdown("### 💾 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    results_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results CSV",
                        results_csv,
                        "model_results.csv",
                        "text/csv"
                    )
                
                with col2:
                    if st.session_state.featured_df is not None:
                        featured_csv = st.session_state.featured_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Featured Data",
                            featured_csv,
                            "featured_data.csv",
                            "text/csv"
                        )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        📊 Question Difficulty Prediction Pipeline | Built with Streamlit<br>
        <small>Upload → Clean → Explore → Engineer → Train → Predict</small>
    </div>
    """,
    unsafe_allow_html=True
)