import streamlit as st
import numpy as np
import xgboost as xgb
import pandas as pd
import datetime
import matplotlib.pyplot as plt

def main():
    html_temp = """
    <div style="background-color: lightblue;padding:16px">
    <h2 style="color:black; text-align:center;">Car Price Prediction Using ML</h2>
    </div>
    """
    model = xgb.XGBRegressor()
    model.load_model(r'C:\Users\Neelam\xgb_model.json')
    
    st.markdown(html_temp, unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    st.markdown("##### Are you planning to sell your car!?\n##### So let's find out the price.")
    p1 = st.number_input("What is the current ex-showroom price of the car(in lakhs)", 2.5, 35.0, step=0.5)
    p2 = st.number_input("What is the distance completed by the car in kilometers?", 100, 500000, step=100)
    
    car_names = [
        "Maruti Suzuki Swift", "Hyundai i20", "Hyundai Creta", "Maruti Suzuki Baleno", 
        "Tata Nexon", "Kia Seltos", "Toyota Innova Crysta", "Honda City", 
        "Renault Kwid", "Hyundai Venue", "Maruti Suzuki Alto", "Tata Tiago", 
        "Mahindra XUV300", "Ford EcoSport", "Honda Amaze", "Hyundai Grand i10 Nios", 
        "Maruti Suzuki Ertiga", "Toyota Glanza", "Volkswagen Polo", "Nissan Magnite", 
        "MG Hector", "Skoda Rapid", "Kia Sonet", "Tata Harrier", 
        "Maruti Suzuki Celerio", "Renault Duster", "Honda WR-V", "Tata Altroz", 
        "Hyundai Verna", "Toyota Yaris", "Mahindra Scorpio", "Ford Figo", 
        "Maruti Suzuki S-Cross", "Hyundai Tucson", "Skoda Kushaq", "MG ZS EV", 
        "Mahindra Thar", "Jeep Compass", "Maruti Suzuki WagonR", "Honda Jazz", 
        "Tata Tigor", "Hyundai Aura", "Nissan Kicks", "Datsun redi-GO", 
        "Volkswagen Vento", "Maruti Suzuki Ignis", "Hyundai Elantra", "Renault Triber", 
        "Kia Carnival", "Tata Safari"
    ]
    
    car_name = st.selectbox("Select the name of the car you want to sell", car_names)
    
    s1 = st.selectbox("What is the fuel type of the car?", ("Petrol", "Diesel", "CNG"))
    if s1 == "Petrol":
        p3 = 0
    elif s1 == "Diesel":
        p3 = 1
    elif s1 == "CNG":
        p3 = 2
    
    s2 = st.selectbox("Are you a dealer or Individual?", ("Dealer", "Individual"))
    if s2 == "Dealer":
        p4 = 0
    elif s2 == "Individual":
        p4 = 1

    s3 = st.selectbox("What is the transmission type?", ("Manual", "Automatic"))
    if s3 == "Manual":
        p5 = 0
    elif s3 == "Automatic":
        p5 = 1 
    
    p6 = st.slider("Number of owners the car previously had?", 0, 3)
    
    date_time = datetime.datetime.now()
    years = st.number_input("In which year the car was purchased", 1990, date_time.year)
    p7 = date_time.year - years
    
    data_new = pd.DataFrame({
        'Present_Price': [p1],
        'Kms_Driven': [p2],
        'Fuel_Type': [p3],
        'Seller_Type': [p4],
        'Transmission': [p5],
        'Owner': [p6],
        'Age': [p7]
    })
    
    if st.button("Predict"):
        pred = model.predict(data_new)
        st.success(f"You can sell your car for {pred[0]:.2f} lakhs")
        
        # For demonstration, using sample actual data and comparing with predicted value
        actual_data = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
        predicted_data = [5.1, 5.6, 5.8, 6.7, 6.9, 7.8, pred[0]]
        
        fig, ax = plt.subplots()
        ax.plot(actual_data, label='Actual Data')
        ax.plot(predicted_data, label='Predicted Data', linestyle='--')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Price (in lakhs)')
        ax.set_title('Actual vs Predicted Car Prices')
        ax.legend()
        
        st.pyplot(fig)

if __name__ == '__main__':
    main()


    
    


    
   