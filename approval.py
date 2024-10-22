import streamlit as st
import pandas as pd
import os
import base64
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Function to upload CSV and store in session_state
def upload_csv():
    uploaded_file = st.file_uploader("📂 Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_df = df  # Store uploaded DataFrame
        # Reset processed DataFrame when a new file is uploaded
        if 'processed_df' in st.session_state:
            del st.session_state.processed_df
        # Reset column selections
        st.session_state.email_column = None
        st.session_state.username_column = None
    return st.session_state.get('uploaded_df', None)

# Function to create conditions for checking empty columns
def check_empty_conditions(df, columns, logical_operator):
    if logical_operator == 'AND':
        return df[columns].notnull().all(axis=1)
    elif logical_operator == 'OR':
        return df[columns].notnull().any(axis=1)
    else:
        return df[columns[0]].notnull()

# Function to prompt OpenAI for approval based on custom criteria
def prompt_openai(question, criteria):
    prompt = f"You are an approval bot. Based on this question or column title: '{question}', determine if this person meets the criteria: '{criteria}' and return either 'Yes' or 'No' and no other words before and after."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an approval bot."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=5,
        temperature=0
    )
    
    result = response.choices[0].message.content.strip()
    return result

# Function to encode CSV to base64
def get_base64_download_link(csv_content, filename):
    b64 = base64.b64encode(csv_content.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    return href

# Initialize session state variables if they don't exist
if 'empty_column_checks' not in st.session_state:
    st.session_state.empty_column_checks = []

if 'open_ended_conditions' not in st.session_state:
    st.session_state.open_ended_conditions = []

# Streamlit Interface
st.title("✅ Automatic Participants Approval")

# Create Tabs
tab1, tab2 = st.tabs(["🔄 Approval Process", "💾 Download Options"])

with tab1:
    st.header("🔄 Approval Process")

    # Step 1: Upload CSV
    df = upload_csv()

    if df is not None:
        st.write("### 📊 Uploaded CSV:")
        st.dataframe(df)

        # Step 2: Select Email and Username Columns
        st.subheader("🔑 Select Key Columns")

        # Email Column Selection
        if st.session_state.email_column is None:
            st.session_state.email_column = st.selectbox(
                "Select the column containing emails:",
                options=df.columns.tolist(),
                key='email_column_selector'
            )
        else:
            st.session_state.email_column = st.selectbox(
                "Select the column containing emails:",
                options=df.columns.tolist(),
                key='email_column_selector',
                index=df.columns.tolist().index(st.session_state.email_column) if st.session_state.email_column in df.columns else 0
            )

        # Username Column Selection
        if st.session_state.username_column is None:
            st.session_state.username_column = st.selectbox(
                "Select the column containing usernames:",
                options=df.columns.tolist(),
                key='username_column_selector'
            )
        else:
            st.session_state.username_column = st.selectbox(
                "Select the column containing usernames:",
                options=df.columns.tolist(),
                key='username_column_selector',
                index=df.columns.tolist().index(st.session_state.username_column) if st.session_state.username_column in df.columns else 0
            )

        st.markdown("---")  # Separator

        # Step 3: Add Empty Column Checks
        st.subheader("🔍 Add Empty Column Checks")

        def add_empty_column_check():
            st.session_state.empty_column_checks.append({'columns': [], 'logical_operator': 'AND'})

        if st.button("➕ Add Empty Column Check", key="add_empty_check"):
            add_empty_column_check()

        # Display existing empty column checks
        for i, condition in enumerate(st.session_state.empty_column_checks):
            st.write(f"**Empty Column Check {i+1}**")
            cols = st.multiselect(
                f"Select columns for Empty Check {i+1}:",
                options=df.columns.tolist(),
                default=condition['columns'],
                key=f"empty_cols_{i}"
            )
            if len(cols) > 1:
                logic = st.radio(
                    f"Logical Operator for Check {i+1}:",
                    options=['AND', 'OR'],
                    index=0 if condition['logical_operator'] == 'AND' else 1,
                    key=f"empty_logic_{i}"
                )
            else:
                logic = 'AND'  # Default logic when only one column is selected

            st.session_state.empty_column_checks[i]['columns'] = cols
            st.session_state.empty_column_checks[i]['logical_operator'] = logic

            if st.button(f"🗑️ Remove Empty Check {i+1}", key=f"remove_empty_check_{i}"):
                st.session_state.empty_column_checks.pop(i)
                st.experimental_rerun()

        st.markdown("---")  # Separator

        # Step 4: Add Open-ended Conditions
        st.subheader("✍️ Add Open-ended Conditions")

        def add_open_ended_condition():
            st.session_state.open_ended_conditions.append({'column': '', 'criteria': ''})

        if st.button("➕ Add Open-ended Condition", key="add_open_condition"):
            add_open_ended_condition()

        # Display existing open-ended conditions
        for i, condition in enumerate(st.session_state.open_ended_conditions):
            st.write(f"**Open-ended Condition {i+1}**")
            col = st.selectbox(
                f"Select column for Condition {i+1}:",
                options=df.columns.tolist(),
                index=df.columns.tolist().index(condition['column']) if condition['column'] in df.columns else 0,
                key=f"open_condition_col_{i}"
            )
            criteria = st.text_input(
                f"Enter criteria for Condition {i+1}:",
                value=condition['criteria'],
                key=f"open_condition_criteria_{i}"
            )

            st.session_state.open_ended_conditions[i]['column'] = col
            st.session_state.open_ended_conditions[i]['criteria'] = criteria

            if st.button(f"🗑️ Remove Open-ended Condition {i+1}", key=f"remove_open_condition_{i}"):
                st.session_state.open_ended_conditions.pop(i)
                st.experimental_rerun()

        st.markdown("---")  # Separator

        # Step 5: Run Approval Process
        st.subheader("🚀 Run Approval Process")

        if st.button("✅ Run Approval", key="run_approval"):
            st.write("### Processing...")
            df['Approval'] = ''  # Initialize Approval column

            # Step 5a: Process Empty Column Checks
            for condition in st.session_state.empty_column_checks:
                cols = condition['columns']
                logic = condition['logical_operator']
                if cols:
                    condition_met = check_empty_conditions(df, cols, logic)
                    df.loc[~condition_met, 'Approval'] = 'No'
                    st.write(f"**Empty Column Check for columns {cols} with logic `{logic}`** applied.")

            # Step 5b: Process Open-ended Conditions
            for condition in st.session_state.open_ended_conditions:
                col = condition['column']
                criteria = condition['criteria']
                if col and criteria:
                    st.write(f"**Processing Open-ended Condition on column '{col}' with criteria '{criteria}'**")
                    for idx, row in df.iterrows():
                        value = row[col]
                        if pd.isnull(value):
                            continue  # Skip if value is NaN
                        if df.at[idx, 'Approval'] == 'No':
                            continue  # Skip already rejected rows
                        result = prompt_openai(value, criteria)
                        if result == 'Yes':
                            df.at[idx, 'Approval'] = 'Yes'
                        elif result == 'No':
                            df.at[idx, 'Approval'] = 'No'
                        # If result is neither 'Yes' nor 'No', leave it blank

            # Step 5c: Store the processed DataFrame
            st.session_state.processed_df = df.copy()

            # Step 5d: Display processed DataFrame
            st.write("### ✅ Approval Process Completed.")
            st.dataframe(st.session_state.processed_df)

with tab2:
    st.header("💾 Download Options")

    if 'processed_df' in st.session_state:
        processed_df = st.session_state.processed_df

        # Generate CSV contents
        csv_full = processed_df.to_csv(index=False)
        ready_approve_df = processed_df[processed_df['Approval'] == 'Yes'][[
            st.session_state.email_column,
            st.session_state.username_column
        ]]
        csv_ready = ready_approve_df.to_csv(index=False)
        manual_check_df = processed_df[~processed_df['Approval'].isin(['Yes', 'No'])]
        csv_manual = manual_check_df.to_csv(index=False)

        # Create download links
        st.markdown("### 📥 Download Full CSV")
        st.markdown(get_base64_download_link(csv_full, "full_processed_file.csv"), unsafe_allow_html=True)

        st.markdown("### ✅ Download Ready to Approve CSV")
        if not ready_approve_df.empty:
            st.markdown(get_base64_download_link(csv_ready, "ready_to_approve.csv"), unsafe_allow_html=True)
        else:
            st.info("There are no approved entries to download.")

        st.markdown("### 🔍 Download Applications Requiring Manual Check")
        if not manual_check_df.empty:
            st.markdown(get_base64_download_link(csv_manual, "manual_check_applications.csv"), unsafe_allow_html=True)
        else:
            st.info("There are no applications requiring manual check.")
    else:
        st.info("🚀 Please run the approval process in the **Approval Process** tab to enable download options.")
        st.download_button("Download Processed CSV", csv, "processed_file.csv", "text/csv")
