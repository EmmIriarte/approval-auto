import streamlit as st
import pandas as pd
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key='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')

def upload_csv():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# Function to create logical conditions
def check_empty_conditions(df, columns, logical_operator):
    if logical_operator == 'AND':
        return df[columns].notnull().all(axis=1)
    elif logical_operator == 'OR':
        return df[columns].notnull().any(axis=1)
    else:
        # No AND/OR selected, we check only one column
        return df[columns[0]].notnull()

# Function to prompt OpenAI for approval based on custom criteria
def prompt_openai(question, criteria):
    prompt = f"You are an approval bot. Based on this question: '{question}', determine if this person meets the criteria: '{criteria}' and return either 'Yes' or 'No' and no other words before and after."
    
    print(f"Prompt sent to OpenAI: {prompt}")  # Log prompt to console

    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an approval bot."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=5,
    temperature=0)
    
    result = response.choices[0].message.content.strip()
    print(f"OpenAI response: {result}")  # Log AI response to console
    return result

# Streamlit Interface
st.title("CSV Logical and Open Conditions Processor")

df = upload_csv()

if df is not None:
    st.write("Uploaded CSV:", df)

    # Logical conditions section
    if 'logical_conditions' not in st.session_state:
        st.session_state.logical_conditions = []

    def add_logical_condition():
        condition = {'columns': [], 'logical_operator': None}
        st.session_state.logical_conditions.append(condition)

    # Add Logical Condition button (Primary)
    if st.button("Add Logical Condition", key="add_logical", help="Add a new logical condition", type="secondary"):
        add_logical_condition()

    # Display existing logical conditions and allow deletion
    for i, condition in enumerate(st.session_state.logical_conditions):
        st.write(f"Logical Condition {i+1}")
        col_selection = st.multiselect(f"Select columns for condition {i+1}", df.columns.tolist(), key=f"col_selection_{i}", default=condition['columns'])
        
        # Enable/disable logical operators based on column selection
        if len(col_selection) > 1:
            logical_operator = st.radio(f"Logical Operator for condition {i+1} (only if selecting multiple columns)", ['AND', 'OR'], index=0 if condition['logical_operator'] == 'AND' else 1, key=f"logic_op_{i}")
        else:
            logical_operator = None
        
        st.session_state.logical_conditions[i]['columns'] = col_selection
        st.session_state.logical_conditions[i]['logical_operator'] = logical_operator if logical_operator else None

        # Remove Logical Condition button (Secondary)
        if st.button(f"Remove Logical Condition {i+1}", key=f"remove_logical_{i}", help="Remove this logical condition", type="primary"):
            st.session_state.logical_conditions.pop(i)
            st.experimental_rerun()

    # Open-ended conditions section
    if 'open_ended_conditions' not in st.session_state:
        st.session_state.open_ended_conditions = []

    def add_open_ended_condition():
        condition = {'column': '', 'criteria': ''}
        st.session_state.open_ended_conditions.append(condition)

    # Add Open-ended Condition button (Primary)
    if st.button("Add Open-ended Condition", key="add_open", help="Add a new open-ended condition", type="secondary"):
        add_open_ended_condition()

    # Display existing open-ended conditions and allow deletion
    for i, condition in enumerate(st.session_state.open_ended_conditions):
        st.write(f"Open-ended Condition {i+1}")
        col = st.selectbox(f"Select column for open-ended condition {i+1}", df.columns.tolist(), key=f"open_col_{i}", index=df.columns.tolist().index(condition['column']) if condition['column'] else 0)
        criteria = st.text_input(f"Describe the acceptable criteria for column {i+1}", key=f"criteria_{i}", value=condition['criteria'])
        st.session_state.open_ended_conditions[i]['column'] = col
        st.session_state.open_ended_conditions[i]['criteria'] = criteria

        # Remove Open-ended Condition button (Secondary)
        if st.button(f"Remove Open-ended Condition {i+1}", key=f"remove_open_{i}", help="Remove this open-ended condition", type="secondary"):
            st.session_state.open_ended_conditions.pop(i)
            st.experimental_rerun()

    # Run button (Primary)
    if st.button("Run", type="secondary"):
        st.write("Processing started")
        df['Approval'] = ''
        
        # Step 1: Process logical conditions
        for condition in st.session_state.logical_conditions:
            cols, logic_op = condition['columns'], condition['logical_operator']
            if cols:
                with st.spinner(f"Processing logical condition for non-empty columns: {cols}"):
                    empty_check = check_empty_conditions(df, cols, logic_op)
                    for index in df[~empty_check].index:
                        df.loc[index, 'Approval'] = 'No'
                        print(f"Row {index}: Logical condition failed, set 'Approval' to 'No'")  # Log to console

        # Step 2: Process open-ended conditions
        for condition in st.session_state.open_ended_conditions:
            col, criteria = condition['column'], condition['criteria']
            if col and criteria:
                with st.spinner(f"Processing open-ended condition for column: {col}"):
                    for index, row in df.iterrows():
                        result = prompt_openai(row[col], criteria)
                        if result == 'Yes':
                            df.at[index, 'Approval'] = 'Yes'
                        print(f"Row {index}: OpenAI result: {result}")  # Log to console
        
        # Step 3: Display processed CSV and provide download link
        st.write("Processing complete.")
        st.write("Processed CSV:", df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Processed CSV", csv, "processed_file.csv", "text/csv")