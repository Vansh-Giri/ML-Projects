import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

# 🔹 Load model, scaler, and columns
model, scaler, model_columns = joblib.load("knn_pipeline.pkl")

# 🔹 Load original dataset
df_raw = pd.read_csv("customer_churn_dataset-training-master.csv")
df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.dropna(subset=["Churn"])
df_raw["Churn"] = df_raw["Churn"].astype(int)

# Clean string columns
for col in ["Gender", "Subscription Type", "Contract Length"]:
    df_raw[col] = df_raw[col].astype(str).str.strip()

# Keep encoded version for model predictions
df = pd.get_dummies(df_raw.drop(columns=["CustomerID"]), columns=["Gender", "Subscription Type", "Contract Length"], drop_first=True)

# Prepare features for similarity
features = model_columns
df_features = df[features].copy()
numeric_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend", "Last Interaction"]
df_features[numeric_cols] = scaler.transform(df_features[numeric_cols])

# 🖥️ Streamlit UI
st.title("Customer Churn Prediction & Plan Recommender")
st.sidebar.header("Customer Info")

inputs = {
    "Age": st.sidebar.number_input("Age", min_value=0),
    "Tenure": st.sidebar.number_input("Tenure", min_value=0),
    "Usage Frequency": st.sidebar.number_input("Usage Frequency", min_value=0),
    "Support Calls": st.sidebar.number_input("Support Calls", min_value=0),
    "Payment Delay": st.sidebar.number_input("Payment Delay", min_value=0),
    "Total Spend": st.sidebar.number_input("Total Spend", min_value=0.0),
    "Last Interaction": st.sidebar.number_input("Last Interaction", min_value=0)
}

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
contract = st.sidebar.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
sub_type = st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])

# Prepare user input
user_df = pd.DataFrame([inputs])
user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])
user_df["Gender_Male"] = gender == "Male"
user_df["Contract Length_Monthly"] = contract == "Monthly"
user_df["Contract Length_Quarterly"] = contract == "Quarterly"
user_df["Subscription Type_Standard"] = sub_type == "Standard"
user_df["Subscription Type_Premium"] = sub_type == "Premium"

for col in model_columns:
    if col not in user_df.columns:
        user_df[col] = False

user_df = user_df[model_columns]

# 🔮 Churn Prediction
if st.sidebar.button("Predict Churn"):
    prediction = model.predict(user_df)[0]
    st.subheader("🔍 Churn Prediction")
    st.write("🟢 Likely to Stay" if prediction == 1 else "🟢 Likely to Stay")

def content_based_recommend():
    retained = df_raw[df_raw["Churn"] == 0].copy()

    top = (
        retained.groupby(["Subscription Type", "Contract Length"])
        .size()
        .reset_index(name="User Count")
    )
    top["User Base (%)"] = (top["User Count"] / len(retained) * 100).round(2)

    # Just rename the readable columns (except User Count)
    top = top.rename(columns={
        "Subscription Type": "Plan",
        "Contract Length": "Contract"
    })

    return top[["Plan", "Contract", "User Base (%)"]].sort_values(by="User Base (%)", ascending=False).head(3)


def knn_based_recommend(user_vector):
    from sklearn.neighbors import NearestNeighbors

    # Fit NearestNeighbors on retained users only
    retained_df = df[df["Churn"] == 0].copy()
    retained_scaled = retained_df[model_columns].copy()
    retained_scaled[numeric_cols] = scaler.transform(retained_scaled[numeric_cols])

    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(retained_scaled)

    distances, indices = knn.kneighbors(user_vector)

    similar_users = df.iloc[indices[0]]

    # Decode plan and contract for each similar user
    def decode_plan(row):
        if row['Subscription Type_Premium']:
            return 'Premium'
        elif row['Subscription Type_Standard']:
            return 'Standard'
        else:
            return 'Basic'

    def decode_contract(row):
        if row['Contract Length_Monthly']:
            return 'Monthly'
        elif row['Contract Length_Quarterly']:
            return 'Quarterly'
        else:
            return 'Annual'

    similar_users['Plan'] = similar_users.apply(decode_plan, axis=1)
    similar_users['Contract'] = similar_users.apply(decode_contract, axis=1)

    top_recommendations = (
        similar_users.groupby(['Plan', 'Contract'])
        .size()
        .reset_index(name='User Base (%)')
        .sort_values(by='User Base (%)', ascending=False)
        .head(3)
    )

    return top_recommendations



# 🧠 Recommend Plans
if st.sidebar.button("Recommend Plans"):
    st.subheader("🎯 Plan Recommendations")

    st.markdown("**✅ Popular Plans Among Retained Customers**")
    st.table(content_based_recommend())

    st.markdown("**✅ Based on Similar Users Who Didn't Churn**")
    st.table(knn_based_recommend())

