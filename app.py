from flask import Flask, render_template, request, redirect, url_for
from flask import Flask, render_template
from io import BytesIO
import base64
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
import pyodbc

matplotlib.use('Agg')  # Ensure we use Agg backend to avoid GUI-related issues
app = Flask(__name__)
SERVER = "nhck.database.windows.net"
DATABASE = "asd"
USERNAME = "nhck3001"
PASSWORD = "Khai@30012003"
connectionString = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}'

try:
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={connectionString}")
    print("Connection successful!")
except Exception as e:
    print(f"Error: {e}")

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        print(f"Username: {username}, Password: {password}, Email: {email}")
        return redirect(url_for('home'))  # Redirect to the home page upon successful login

    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/household')
def household_data():
    query = """
 SELECT t.hshd_num, t.basket_num, t.PURCHASE, t.product_num, p.department, p.commodity, t.SPEND, t.UNITS,t.STORE_R,t.WEEK_NUM,t.[YEAR],h.L, h.AGE_RANGE, h.MARITAL, h.INCOME_RANGE, h.HOMEOWNER, h.HSHD_COMPOSITION, h.HH_SIZE, h.CHILDREN
    FROM transactions t
    JOIN households h ON t.hshd_num = h.hshd_num
    JOIN products p ON t.product_num = p.product_num
    WHERE t.hshd_num = 10
    ORDER BY t.hshd_num, t.basket_num, t.PURCHASE
    """
    # Fetch data from SQL
    df = pd.read_sql(query, engine)
    # Render the data into an HTML table (using Jinja template)
    return render_template('household.html', tables=df.to_html(classes='data', index=False, border=0, justify='center'))


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        hshd_num = request.form['hshd_num']  # Get the hshd_num from the form
        query = """  
        SELECT t.hshd_num, t.basket_num, t.PURCHASE, t.product_num, p.department, p.commodity, t.SPEND, t.UNITS,t.STORE_R,t.WEEK_NUM,t.[YEAR],h.L, h.AGE_RANGE, h.MARITAL, h.INCOME_RANGE
        FROM transactions t
        JOIN households h ON t.hshd_num = h.hshd_num
        JOIN products p ON t.product_num = p.product_num
        WHERE t.hshd_num = ?
        ORDER BY t.hshd_num, t.basket_num, t.PURCHASE
        """
    
        # Fetch data from SQL based on hshd_num  
        df = pd.read_sql(query, engine, params=(hshd_num,))  # Pass as tuple (hshd_num,)
        return render_template('household.html', tables=df.to_html(classes='data', index=False, border=0, justify='center'))
    
    return render_template('search.html') 

@app.route('/dashboard')
def dashboard():
    # SQL query to retrieve data for the analysis: how income affects total spending
    query = """
        SELECT h.INCOME_RANGE, SUM(t.SPEND) AS total_spend
        FROM households h
        JOIN transactions t ON h.hshd_num = t.hshd_num
        GROUP BY h.INCOME_RANGE
        ORDER BY h.INCOME_RANGE
    """

    # Fetch data from the database into a DataFrame
    df = pd.read_sql(query, engine)
    df['INCOME_RANGE'] = df['INCOME_RANGE'].str.strip()
    df = df.groupby('INCOME_RANGE')['total_spend'].sum().reset_index()
    # Handle None values in INCOME_RANGE by replacing them with a placeholder (e.g., "Unknown")
    df['INCOME_RANGE'] = df['INCOME_RANGE'].fillna('Unknown')
    # Ensure we sort the data by INCOME_RANGE for proper plotting order
    df = df.sort_values('INCOME_RANGE')
    # Create a bar plot to show the relationship between income range and total spending
    plt.figure(figsize=(10, 6))
    plt.bar(df['INCOME_RANGE'], df['total_spend'], color='skyblue')
    plt.title('Impact of Income on Total Spend')
    plt.xlabel('Income Range')
    plt.ylabel('Total Spend')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Convert the plot to PNG and then to base64 to embed it in HTML
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_income = base64.b64encode(img.getvalue()).decode('utf8')

    # National vs private brand graph
    # SQL query to retrieve data for private vs national brands and organic items preferences
    query_brand_organic = """
        SELECT p.BRAND_TY, p.NATURAL_ORGANIC_FLAG, SUM(t.SPEND) AS total_spend
        FROM transactions t
        JOIN products p ON t.product_num = p.product_num
        GROUP BY p.BRAND_TY, p.NATURAL_ORGANIC_FLAG
        ORDER BY p.BRAND_TY, p.NATURAL_ORGANIC_FLAG
    """

    # Fetch the data into a DataFrame
    df_brand_organic = pd.read_sql(query_brand_organic, engine)

    # Handle missing values
    df_brand_organic['BRAND_TY'] = df_brand_organic['BRAND_TY'].fillna('Unknown')
    df_brand_organic['NATURAL_ORGANIC_FLAG'] = df_brand_organic['NATURAL_ORGANIC_FLAG'].fillna('Non-Organic')

    # Create a bar plot for brand and organic flag preferences
    plt.figure(figsize=(10, 6))
    df_brand_organic_grouped = df_brand_organic.groupby(['BRAND_TY', 'NATURAL_ORGANIC_FLAG'])['total_spend'].sum().unstack()
    df_brand_organic_grouped.plot(kind='bar', stacked=True, color=['lightgreen', 'lightblue'], ax=plt.gca())
    plt.title('Brand Type and Organic Item Preferences')
    plt.xlabel('Brand Type')
    plt.ylabel('Total Spend')
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Convert the plot to PNG and then to base64 to embed it in HTML
    img_brand_organic = BytesIO()
    plt.savefig(img_brand_organic, format='png')
    img_brand_organic.seek(0)
    plot_url_brand_organic = base64.b64encode(img_brand_organic.getvalue()).decode('utf8')

        # SQL query to retrieve total spend by week for each year
    query_spend_by_week = """
        SELECT t.YEAR, t.WEEK_NUM, SUM(t.SPEND) AS total_spend
        FROM transactions t
        GROUP BY t.YEAR, t.WEEK_NUM
        ORDER BY t.YEAR, t.WEEK_NUM
    """

    # Fetch data from the database into a DataFrame
    df_spend_by_week = pd.read_sql(query_spend_by_week, engine)

    # Ensure 'YEAR' and 'WEEK_NUM' are treated as categorical
    df_spend_by_week['YEAR'] = df_spend_by_week['YEAR'].astype(str)

    # Create a line plot to show total spend by week for each year
    plt.figure(figsize=(10, 6))

    # Loop through each year and plot a line for each year
    for year in df_spend_by_week['YEAR'].unique():
        print(df_spend_by_week['YEAR'].unique())
        if (year != "2018"):
            year_data = df_spend_by_week[df_spend_by_week['YEAR'] == year]
            plt.plot(year_data['WEEK_NUM'], year_data['total_spend'], label=f'Year {year}')

    # Adding labels and title
    plt.title('Total Spend by Week for Each Year')
    plt.xlabel('Week Number')
    plt.ylabel('Total Spend')
    plt.legend(title='Year')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convert the plot to PNG and then to base64 to embed it in HTML
    img_spend_by_week = BytesIO()
    plt.savefig(img_spend_by_week, format='png')
    img_spend_by_week.seek(0)
    plot_url_spend_by_week = base64.b64encode(img_spend_by_week.getvalue()).decode('utf8')


    # SQL query to retrieve total spend by commodity, week, and year
    query_product_popularity = """
        SELECT p.commodity, t.YEAR, t.WEEK_NUM, SUM(t.SPEND) AS total_spend
        FROM transactions t
        JOIN products p ON t.product_num = p.product_num
        GROUP BY p.commodity, t.YEAR, t.WEEK_NUM
        ORDER BY t.YEAR, t.WEEK_NUM, p.commodity
    """

    # Fetch data from the database into a DataFrame
    df = pd.read_sql(query_product_popularity, engine)

    # Convert YEAR and WEEK_NUM into a datetime for easy plotting
    df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + df['WEEK_NUM'].astype(str) + '0', format='%Y%U%w')

    # Create a plot for product popularity over time
    plt.figure(figsize=(12, 6))
    
    # Iterate through each commodity and plot its total spend over time
    for commodity in df['commodity'].unique():
        commodity_df = df[df['commodity'] == commodity]
        plt.plot(commodity_df['Date'], commodity_df['total_spend'], label=commodity)

    plt.title('Product Category Popularity Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Spend')
    plt.legend(title='Commodity', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convert the plot to PNG and then to base64 to embed it in HTML
    img_popularity = BytesIO()
    plt.savefig(img_popularity, format='png')
    img_popularity.seek(0)
    plot_url_popularity = base64.b64encode(img_popularity.getvalue()).decode('utf8')

    # Return the dashboard page with all the plots
    return render_template('dashboard.html', 
                           plot_url_income=plot_url_income,
                           plot_url_brand_organic=plot_url_brand_organic,
                           plot_url_3=plot_url_spend_by_week,
                           plot_url_4=plot_url_popularity)


@app.route('/cross-selling')
def cross_selling_types():
    # Step 1: Get basket + commodity data
    query = """
        SELECT TOP 10000 t.basket_num, p.commodity
        FROM transactions t
        JOIN products p ON t.product_num = p.product_num
        WHERE p.commodity IS NOT NULL
        ORDER BY t.basket_num
    """
    df = pd.read_sql(query, engine)

    # Step 2: Basket-Commodity matrix
    basket = df.groupby(['basket_num', 'commodity']).size().unstack(fill_value=0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Step 3: Top 5 commodities by frequency
    top_commodities = basket.sum().sort_values(ascending=False).head(5).index

    results = []

    for target in top_commodities:
        X = basket.drop(columns=target)
        y = basket[target]

        if y.sum() < 10:
            continue  # Skip rare classes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        importances = model.feature_importances_
        top_features = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(3)

        results.append({
            "target_type": target,
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1-score": report["1"]["f1-score"],
            "top_cross_sell": top_features.to_dict()
        })

    # Step 4: Print cross-sell results
    print("\n🔍 Top Cross-Sell Commodity Relationships:\n")
    for result in results:
        print(f"🎯 Target Type: {result['target_type']}")
        print(f"   Precision: {result['precision']:.2f} | Recall: {result['recall']:.2f} | F1-score: {result['f1-score']:.2f}")
        print("   Top Predictive Types:")
        for pred, score in result['top_cross_sell'].items():
            print(f"     - {pred} (Importance: {score:.4f})")
        print("-" * 60)

    # Step 5: Prepare plot
    plot_data = []
    for result in results:
        for pred, imp in result['top_cross_sell'].items():
            plot_data.append({
                'Target': result['target_type'],
                'Predictor': pred,
                'Importance': imp
            })
    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 6))
    for target in plot_df['Target'].unique():
        subset = plot_df[plot_df['Target'] == target]
        labels = [f"{target} → {pred}" for pred in subset['Predictor']]
        plt.bar(labels, subset['Importance'], label=target)

    plt.xlabel("Predictor → Target Type")
    plt.ylabel("Feature Importance")
    plt.title("Top 3 Cross-Sell Commodity Type Predictors")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Target Commodity Type")
    plt.tight_layout()

    # Convert to base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('cross_selling.html', plot_url=plot_url)

@app.route("/disengage")
def disengage():
    # SQL query to load actual transaction and household data (non-loyal only)
    query = '''
        SELECT TOP 10000 t.HSHD_NUM, t.WEEK_NUM, t.YEAR, t.SPEND, h.L, h.AGE_RANGE, h.MARITAL, h.INCOME_RANGE, 
               h.HSHD_COMPOSITION, h.CHILDREN
        FROM transactions t
        JOIN households h ON t.HSHD_NUM = h.HSHD_NUM
        WHERE h.L = 'N'
'''

    
    df = pd.read_sql(query, engine)

    df['date'] = df['YEAR'] * 52 + df['WEEK_NUM']
    weekly = df.groupby(['HSHD_NUM', 'date'])['SPEND'].sum().reset_index()

    def calc_trend(sub):
        X = sub['date'].values.reshape(-1, 1)
        y = sub['SPEND'].values
        if len(X) < 3:
            return 0
        model = LinearRegression().fit(X, y)
        return model.coef_[0]

    trend = weekly.groupby('HSHD_NUM', group_keys=False).apply(calc_trend).reset_index()
    trend.columns = ['HSHD_NUM', 'slope']
    trend['DISENGAGED'] = (trend['slope'] < 0).astype(int)

    features = df.drop_duplicates('HSHD_NUM')[[
        'HSHD_NUM', 'AGE_RANGE', 'MARITAL', 'INCOME_RANGE', 'HSHD_COMPOSITION', 'CHILDREN']]
    merged = trend.merge(features, on='HSHD_NUM')

    # Prepare features for logistic regression
    X = pd.get_dummies(merged.drop(columns=['HSHD_NUM', 'slope', 'DISENGAGED']), drop_first=True)
    y = merged['DISENGAGED']

    # Remove constant and collinear columns using VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    def calculate_vif(df):
        vif_data = pd.DataFrame()
        vif_data['feature'] = df.columns
        vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        return vif_data

    X = sm.add_constant(X)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    X = X.astype(np.float64)
    y = y.astype(np.float64)

    # Drop features with high VIF
    while True:
        vif = calculate_vif(X.drop(columns=['const'], errors='ignore'))
        max_vif = vif.loc[vif['VIF'] > 10]
        if max_vif.empty:
            break
        drop_col = max_vif.sort_values('VIF', ascending=False)['feature'].iloc[0]
        X = X.drop(columns=[drop_col])

    # Remove constant columns and multicollinearity using variance threshold
    nunique = X.nunique()
    X = X.loc[:, nunique > 1]
    X = sm.add_constant(X)

    # Drop NaNs/Infs
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    # Check if matrix is invertible
    try:
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        model = sm.Logit(y, X).fit(disp=0)
        regression_summary = model.summary2().as_text()
    except Exception as e:
        regression_summary = f"Logistic regression failed due to a matrix issue.\n\nError: {e}\n\nTry reducing correlated variables or inspecting the dummies."
        plot_url = None
        return render_template("disengage.html", plot_url=plot_url, regression_summary=regression_summary)

    # Visualization
    plt.figure(figsize=(8, 5))
    sns.barplot(data=merged, x='AGE_RANGE', y='DISENGAGED', estimator=np.mean)
    plt.title("Disengagement Rate by Age Range")
    plt.ylabel("Proportion Disengaged")
    plt.xticks(rotation=45)
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template("disengage.html", plot_url=plot_url, regression_summary=regression_summary)

if __name__ == '__main__':
    app.run(port = 8000,debug=True)
