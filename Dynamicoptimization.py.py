import streamlit as st
import pandas as pd
import pulp
import numpy as np

# Optimization function
def optimize_supplier(df, nego_percent, total_demand, target_quality, target_delivery):
    df = df.copy()
    df["EffectivePrice"] = df["UnitPrice"] * (1 - np.array(nego_percent) / 100)
    if total_demand > df["MaxCapacity"].sum():
        return df.assign(OptimizedVolume=0, OptimizedSpend=0, Savings=0), "Infeasible (Demand > Total Capacity)"
    model = pulp.LpProblem("Supplier_Optimization", pulp.LpMinimize)
    volumes = {
        row.Supplier: pulp.LpVariable(
            f"vol_{row.Supplier}",
            lowBound=0,
            upBound=int(row.MaxCapacity),
            cat="Integer"
        )
        for _, row in df.iterrows()
    }
    model += pulp.lpSum(volumes[row.Supplier] * row.EffectivePrice for _, row in df.iterrows())
    model += pulp.lpSum(volumes[row.Supplier] for _, row in df.iterrows()) == int(total_demand)
    model += pulp.lpSum(volumes[row.Supplier] * row.Quality for _, row in df.iterrows()) >= target_quality * total_demand
    model += pulp.lpSum(volumes[row.Supplier] * row.Delivery for _, row in df.iterrows()) >= target_delivery * total_demand
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[model.status]
    if status == "Optimal":
        df["OptimizedVolume"] = [int(pulp.value(volumes[row.Supplier])) for _, row in df.iterrows()]
        df["OptimizedSpend"] = df["OptimizedVolume"] * df["EffectivePrice"]
        df["Savings"] = df["CurrentVolume"] * df["UnitPrice"] - df["OptimizedSpend"]
    else:
        df["OptimizedVolume"] = 0
        df["OptimizedSpend"] = 0
        df["Savings"] = 0
    return df, status

# Streamlit UI
def main():
    st.title("Supplier Optimization Dashboard")
    data = {
        "Supplier": ["S1", "S2"],
        "UnitPrice": [3.365, 2.830],
        "CurrentVolume": [1060527, 799717],
        "Quality": [0.95, 0.92],
        "Delivery": [0.93, 0.89],
        "MaxCapacity": [12000, 900]
    }
    df = pd.DataFrame(data)
    total_demand = st.number_input("Total Demand", value=18600, step=100)
    target_quality = st.number_input("Min Quality", value=0.8, step=0.05)
    target_delivery = st.number_input("Min Delivery", value=0.8, step=0.05)
    nego_s1 = st.slider("S1 %", -20.0, 20.0, 0.0)
    nego_s2 = st.slider("S2 %", -20.0, 20.0, 0.0)
    s1_cap = st.number_input("S1 Max Cap", value=12000, step=100)
    s2_cap = st.number_input("S2 Max Cap", value=900, step=100)
    df.loc[df["Supplier"] == "S1", "MaxCapacity"] = s1_cap
    df.loc[df["Supplier"] == "S2", "MaxCapacity"] = s2_cap
    if st.button("Run Optimization"):
        nego_percent = [nego_s1, nego_s2]
        results, status = optimize_supplier(df, nego_percent, total_demand, target_quality, target_delivery)
        st.subheader("=== Optimization Results ===")
        st.write(f"Solver Status: {status}")
        st.dataframe(results)
        if status == "Optimal":
            total_quality = (results["OptimizedVolume"] * results["Quality"]).sum() / total_demand
            total_delivery = (results["OptimizedVolume"] * results["Delivery"]).sum() / total_demand
            total_spend = results["OptimizedSpend"].sum()
            total_savings = results["Savings"].sum()
            st.write(f"Achieved Quality: {total_quality:.2f}")
            st.write(f"Achieved Delivery: {total_delivery:.2f}")
            st.write(f"Total Spend: ₹{total_spend:,.2f}")
            st.write(f"Total Savings: ₹{total_savings:,.2f}")

if __name__ == "__main__":
    main()
