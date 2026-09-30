"""
Run: $ python -m streamlit run app_streamlit.py
"""

from __future__ import annotations

import json
import boto3
import pandas as pd
import streamlit as st

# Tab config
st.set_page_config(
    page_title="Credit Score Classifier",
    page_icon="💳",
    layout="centered",
)

SAGEMAKER_ENDPOINT_NAME = "CreditScore-endpoint"

TARGET_NAMES = {0: "Poor", 1: "Standard", 2: "Good"}
TARGET_COLOR = {"Poor": "🔴", "Standard": "🟡", "Good": "🟢"}

CREDIT_MIX_OPTIONS = ["Unknown", "Bad", "Standard", "Good"]
PAYMENT_MIN_AMOUNT_OPTIONS = ["Unknown", "No", "Yes"]
PAYMENT_BEHAVIOUR_OPTIONS = [
    "Low_spent_Small_value_payments",
    "Low_spent_Medium_value_payments",
    "Low_spent_Large_value_payments",
    "High_spent_Small_value_payments",
    "High_spent_Medium_value_payments",
    "High_spent_Large_value_payments",
    "Unknown",
]
LOAN_TYPE_OPTIONS = [
    "Not Specified", "Personal Loan", "Auto Loan", "Home Equity Loan",
    "Mortgage Loan", "Credit-Builder Loan", "Debt Consolidation Loan",
    "Payday Loan", "Student Loan",
]

ALL_MODEL_COLS = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance",
    "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour",
    "Num_Type_of_Loan", "Credit_History_Age_Months",
]

# SageMaker client — EC2 otomatis menggunakan IAM Role yang di-attach, tidak perlu hardcode credentials
@st.cache_resource  # fitur dari streamlit untuk simpan client di RAM Streamlit
def load_client():
    try:
        client = boto3.client("sagemaker-runtime")
        return client, None
    except Exception as e:
        return None, str(e)

# Fungsi untuk predict melalui endpoint SageMaker
def predict(client, values: dict):
    """Mengirim 1 row data ke SageMaker endpoint dan return label
    prediksi serta dictionary berisi probabilitas untuk setiap kelas.
    """
    payload = json.dumps({"instances": [{col: values[col] for col in ALL_MODEL_COLS}]})

    # Memanggil SageMaker endpoint
    response = client.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )

    # Membaca response dari endpoint
    result = json.loads(response["Body"].read().decode("utf-8"))

    label = result["labels"][0]
    proba_dict = {name: result["probabilities"][0][i] for i, name in TARGET_NAMES.items()}
    return label, proba_dict


# Sidebar
with st.sidebar:
    st.title("💳 Credit Score Classifier")
    st.markdown(
        "Classifier ini menggunakan model dengan **Macro F1-Score** terbaik saat EDA yaitu model **LightGBM**"
    )
    st.divider()

    client, load_error = load_client()
    if client is not None:
        st.success("SageMaker endpoint connected")
        st.caption(f"`{SAGEMAKER_ENDPOINT_NAME}`")
    else:
        st.error("Gagal terhubung ke SageMaker")
        st.code(load_error)

# Header
st.title("Prediksi Credit Score")
st.caption(
    "Masukkan data profil finansial nasabah pada form di bawah untuk mendapatkan prediksi skor kredit dalam kategori Poor, Standard, atau Good."
)

if client is None:
    st.warning("Fix error dari load client sebelum ke tahap berikutnya.")
    st.stop()


# Input form
st.subheader("Profil Nasabah")

with st.form("prediction_form"):

    st.markdown("**👤Pendapatan & Demografi**")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with c2:
        annual_income = st.number_input(
            "Annual Income", min_value=0.0, value=50000.0, step=1000.0
        )
    with c3:
        monthly_inhand_salary = st.number_input(
            "Monthly Inhand Salary", min_value=0.0, value=4000.0, step=100.0
        )

    st.markdown("**💳 Akun Bank & Kartu Kredit**")
    c1, c2, c3 = st.columns(3)
    with c1:
        num_bank_accounts = st.number_input(
            "Num Bank Accounts", min_value=0, max_value=20, value=3
        )
    with c2:
        num_credit_card = st.number_input(
            "Num Credit Card", min_value=0, max_value=20, value=3
        )
    with c3:
        interest_rate = st.number_input(
            "Interest Rate (%)", min_value=0, max_value=100, value=12
        )

    st.markdown("**📑 Pinjaman & Riwayat Pembayaran**")
    c1, c2, c3 = st.columns(3)
    with c1:
        num_of_loan = st.number_input(
            "Num of Loan", min_value=0, max_value=20, value=2
        )
    with c2:
        num_type_of_loan = st.number_input(
            "Num Type of Loan (distinct types)",
            min_value = 0, max_value = 10, value = 2 ,
            help = "Jumlah kategori pinjaman yang berbeda yang dimiliki oleh nasabah.",
        )
    with c3:
        credit_history_age_months = st.number_input(
            "Credit History Age (Bulan)",
            min_value=0, max_value=600, value=180,
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        delay_from_due_date = st.number_input(
            "Delay from Due Date (days)", min_value=0, max_value=120, value=14
        )
    with c2:
        num_of_delayed_payment = st.number_input(
            "Num of Delayed Payment", min_value=0, max_value=100, value=5
        )
    with c3:
        num_credit_inquiries = st.number_input(
            "Num Credit Inquiries", min_value=0, max_value=50, value=3
        )

    st.markdown("**💰 Kondisi Finansial**")
    c1, c2, c3 = st.columns(3)
    with c1:
        changed_credit_limit = st.number_input(
            "Changed Credit Limit", min_value=0.0, value=10.0, step=0.5
        )
    with c2:
        outstanding_debt = st.number_input(
            "Outstanding Debt", min_value=0.0, value=1200.0, step=50.0
        )
    with c3:
        credit_utilization_ratio = st.number_input(
            "Credit Utilization Ratio (%)",
            min_value=0.0, max_value=100.0, value=30.0, step=0.5,
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        total_emi_per_month = st.number_input(
            "Total EMI per Month", min_value=0.0, value=150.0, step=10.0
        )
    with c2:
        amount_invested_monthly = st.number_input(
            "Amount Invested Monthly", min_value=0.0, value=200.0, step=10.0
        )
    with c3:
        monthly_balance = st.number_input(
            "Monthly Balance", min_value=0.0, value=300.0, step=10.0
        )

    st.markdown("**🧠 Profil Kredit & Perilaku Pembayaran**")
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_mix = st.selectbox(
            "Credit Mix", CREDIT_MIX_OPTIONS,
            index=2,
            help=(
                '''Pilih **Unknown** jika jenis kredit nasabah tidak diketahui atau tidak tercatat.
                Kategori ini bukan sekadar data kosong. Dalam Training, sekitar 20% nasabah berada di kategori ini. 
                Polanya juga berbeda dari Bad, Standard, atau Good, sehingga model memperlakukannya sebagai kategori tersendiri.'''
            ),
        )
    with c2:
        payment_of_min_amount = st.selectbox(
            "Payment of Min Amount", PAYMENT_MIN_AMOUNT_OPTIONS,
            index=1,
            help=(
					''' Pilih **Unknown** jika tidak ada catatan apakah nasabah membayar minimum amount tagihan.
                    Sekitar 12% data latih berada pada kategori ini dan menunjukkan 
                    pola risiko yang berbeda dari "Yes" maupun "No", sehingga tetap
                    dianggap sebagai opsi yang valid.'''
            ),
        )
    with c3:
        payment_behaviour = st.selectbox(
            "Payment Behaviour", PAYMENT_BEHAVIOUR_OPTIONS,
            index=1,
            help=(
    			'''Pilih **Unknown** jika perilaku pembayaran (besaran dan frekuensi belanja) nasabah
    			tidak dapat ditentukan dari data yang tersedia.''',
            ),
        )

    submitted = st.form_submit_button("Prediksi Credit Score", use_container_width=True)


# Prediction result
if submitted:
    values = {
        "Age": age,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_inhand_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_of_loan,
        "Delay_from_due_date": delay_from_due_date,
        "Num_of_Delayed_Payment": num_of_delayed_payment,
        "Changed_Credit_Limit": changed_credit_limit,
        "Num_Credit_Inquiries": num_credit_inquiries,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization_ratio,
        "Total_EMI_per_month": total_emi_per_month,
        "Amount_invested_monthly": amount_invested_monthly,
        "Monthly_Balance": monthly_balance,
        "Credit_Mix": credit_mix,
        "Payment_of_Min_Amount": payment_of_min_amount,
        "Payment_Behaviour": payment_behaviour,
        "Num_Type_of_Loan": num_type_of_loan,
        "Credit_History_Age_Months": credit_history_age_months,
    }

    try:
        label, proba = predict(client, values)

        st.divider()
        st.subheader("Hasil Prediksi")

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Predicted Credit Score", f"{TARGET_COLOR.get(label, '')} {label}")
        with col_b:
            proba_df = pd.DataFrame({
                "Class": list(proba.keys()),
                "Probability": list(proba.values()),
            }).set_index("Class")
            st.bar_chart(proba_df)

	#Error Handling
    except Exception as e:
        st.error(f"Prediction failed: {e}")