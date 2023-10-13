#!/usr/bin/env python
# coding: utf-8

# # Glass Identification Dataset Data Cleaning and Preprocessing

# # 1. Load Dataset

# In[1]:


import pandas as pd

df = pd.read_excel('Glass_Identification_Data.xlsx')


# #  1. a Print summary statistics:

# In[2]:


#Summary statistics of the above dataset
df.describe()


# # 1. b Summary for all columns of the DataFrame

# In[3]:


#Describing all columns regardless of data type
df.describe(include='all')  


# # 2. Percentage of null/missing values for each variable

# In[4]:


#Calculate the missing values percentage
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
missing_value_df


# # 3. Drop the variables which have more than 75% missing values

# In[5]:


#Get only the columns that have less than 75% missing values
df = df[df.columns[df.isnull().mean() < 0.75]]
df


# # 4. Impute more than 10 missing value columns

# In[6]:


#Get count of all missing values from each column
sum_miss_values = df.isnull().sum()
sum_miss_values


# In[7]:


def impute_missing_values(df):
    # Calculate the mean values for each attribute per class
    class_means = df.groupby("Class").transform("mean")
    
    # Loop on each attribute and impute missing values with class means
    for col in df.columns[1:-1]:  # exclude ID and Class columns
        missing_values = df[col].isnull()
        if missing_values.sum() > 10:
            df.loc[missing_values, col] = class_means.loc[missing_values, col]
    return df
df = impute_missing_values(df)
df


# In[8]:


#check again to verify if there are any columns having more than 10 missing values
sum_miss_values = df.isnull().sum()
sum_miss_values


# # 5. Impute variable containing 10 or lesser than 10 missing records

# In[9]:


#Calculate new class means
class_means = df.groupby('Class').mean()

#Define a function to impute missing values using the previous non-NaN value within the same class
def impute_missing_values(row, attribute):
    if pd.isnull(row[attribute]):
        class_mean = class_means.at[row['Class'], attribute]
        if pd.notnull(class_mean):
            return class_mean
        else:
            return row[attribute]  # if no previous non-NaN value in the same class return NaN
    else:
        return row[attribute]  #if no missing value return the original value

#Get the list of attributes to impute excluding ID and Class 
attributes_to_impute = df.columns.difference(['ID', 'Class'])

#Impute missing values for each attribute
for attribute in attributes_to_impute:
    df[attribute] = df.apply(impute_missing_values, axis=1, args=(attribute,))

df


# # 6. Check if all the missing values are handled

# In[10]:


#check again to verify if there are any columns having 10 or lesser missing values
sum_miss_values = df.isnull().sum()
sum_miss_values


# # 7. Bar Chart

# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create bar chart to visualize the distribution of the 'Class' column
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Class', palette='Set1')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Distribution of Glass Classes')
plt.xticks(rotation=45)  

# Show the plot
plt.tight_layout()
plt.show()


# # 8. Correlation Matrix

# In[12]:


#Excluding 'ID' and 'Class'
predictors_columns = df.drop(columns=['ID', 'Class'])

#Compute the correlation matrix
corr_matrix = predictors_columns.corr()

#Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Predictors')
plt.show()


# In the heatmap, the areas where the color is closer to 1 the features that are highly positively correlated. Features that are not correlated will have correlation values close to 0. 
# Areas where the color is closer to -1 shows features that are highly negatively correlated.

# # 9. Histograms

# In[13]:


#List to store modes
modes = []

#Create subplots for each histogram
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
axes = axes.flatten()

#Iterate for numerical predictors and plot histograms
for i, column in enumerate(predictors_columns.columns):
    ax = axes[i]
    ax.hist(predictors_columns[column], bins=15, edgecolor='k', alpha=0.7)
    ax.set_title(column)
    
    # Calculate and store the mode
    mode = predictors_columns[column].mode().values[0]
    modes.append((column, mode))
    
#Display the histograms
plt.tight_layout()
plt.show()


# # a. Modes of each histogram

# In[14]:


print("Modes for each histogram:")
for column, mode in modes:
    print(f"{column}: {mode}")


# # b. Comparative Analysis of RI and AI

# In[15]:


predictors_columns = df.drop(columns=['ID','Na','Mg','Si','K','CA','Fe','Class'])

#Create subplots for each histogram
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 10))
axes = axes.flatten()

#Iterate for numerical predictors and plot histograms
for i, column in enumerate(predictors_columns.columns):
    ax = axes[i]
    ax.hist(predictors_columns[column], bins=15, edgecolor='k', alpha=0.7)
    ax.set_title(column)
    
    # Calculate and store the mode
    mode = predictors_columns[column].mode().values[0]
    modes.append((column, mode))
    
#Display the histograms
plt.tight_layout()
plt.show()


# RI:
# Central Tendency: The mode of the 'RI' histogram is approximately 1.52, which is the most frequently occurring value.
# Shape: The 'RI' histogram is unimodal, indicating one central peak.
# Skewness: The distribution appears slightly negatively skewed, as it has a longer tail on the left (towards lower values).
# Spread: The values are concentrated around the mode, and the spread of the data is relatively narrow.
#     
# AI:
# Central Tendency: The mode of the 'Al' histogram is approximately 1.0, which is the most frequently occurring aluminum content.
# Shape: The 'Al' histogram is also unimodal, with one central peak.
# Skewness: The distribution appears slightly positively skewed, with a longer tail on the right (towards higher values).
# Spread: The values are concentrated around the mode, but there is more variation in aluminum content compared to the 'RI' distribution.

# # 10. Boxplot

# In[16]:


predictors_columns = df.drop(columns=['ID', 'Class'])

# Create subplots for each boxplot
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
axes = axes.flatten()

# List to store the summaries for each plot
summaries = []

# Iterate over numerical predictors and plot boxplots
for i, column in enumerate(predictors_columns.columns):
    ax = axes[i]
    sns.boxplot(data=df, y=column, ax=ax)
    ax.set_title(column)
    
    # Calculate the five-number summary: min, 25%, median, 75%, max
    summary = df[column].describe()[['min', '25%', '50%', '75%', 'max']]
    summaries.append((column, summary))

# Display the boxplots
plt.tight_layout()
plt.show()


# # Comparitive analysis of RI and AI

# In[17]:


# Select 2 features for comparative analysis 'RI' and 'Al'
feature1 = 'RI'
feature2 = 'Al'

# Find and display the summaries for the selected features
summary1 = next(item[1] for item in summaries if item[0] == feature1)
summary2 = next(item[1] for item in summaries if item[0] == feature2)

print(f"Five-Number Summary for {feature1}:")
print(summary1)
print(f"Five-Number Summary for {feature2}:")
print(summary2)


# # Forest Fires Dataset Data Visualization

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Transform csv file format to pandas DataFrame
df = pd.read_csv("forestfires.csv") 


# # 1. Stacked Bar Chart:

# In[19]:


# Create a new column to extract the month from the 'month' column and order it chronologically
month_order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)

# Group the data by month and day of the week and calculate the count
grouped_data = df.groupby(['month', 'day']).size().unstack(fill_value=0)

# Create a stacked bar chart
ax = grouped_data.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title("Forest Fires by Month and Day of the Week")
plt.xlabel("Month")
plt.ylabel("Number of Fires")
plt.xticks(rotation=45, ha="right")

# Show the legend outside of the plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Show the plot
plt.tight_layout()
plt.show()


# Stacked bar charts are suitable for showing the composition of a whole, but they are less effective for comparing individual categories across different groups like comparing the number of fires on a specific day across months. To rectify this we can visualize using grouped bar chart

# # Rectification: Grouped Bar Chart

# In[20]:


df = pd.read_csv("forestfires.csv") 

# Create a new column to extract the month from the 'month' column and order it chronologically
month_order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)

# Group the data by month and day of the week and calculate the count
grouped_data = df.groupby(['month', 'day']).size().unstack(fill_value=0)

# Get the list of months and days for the x-axis
months = grouped_data.index
days = grouped_data.columns

# Set the width of each bar
bar_width = 0.2

# Create an array of x values for each group of bars
x = np.arange(len(months))

# Create the grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Iterate through the days of the week and plot bars for each day
for i, day in enumerate(days):
    ax.bar(x + i * bar_width, grouped_data[day], width=bar_width, label=day)

# Set the x-axis labels to be the months
ax.set_xticks(x + bar_width * (len(days) - 1) / 2)
ax.set_xticklabels(months, rotation=45, ha="right")

# Set labels and title
plt.xlabel("Month")
plt.ylabel("Number of Fires")
plt.title("Forest Fires by Month and Day of the Week")

# Show the legend
plt.legend(title="Day of the Week")

# Show the plot
plt.tight_layout()
plt.show()


# # 2. Heatmap:

# In[21]:


#Select required columns from the DataFrame
df_corr = df[['area','wind','temp']]

#Calculate the correlation matrix
corr_matrix = df_corr.corr()

#Create a heatmap of the correlation coefficients
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation heatmap of area, wind, and temp variables')

#Show the plot
plt.show()


# # 3. Joint Distribution:

# In[22]:


#Create a joint distribution of wind and area
sns.jointplot(x='wind', y='area', data=df, height=10)
plt.title('Joint distribution of wind and area')

#Show the plot
plt.show()


# In[23]:


#Create a joint distribution of temp and area
sns.jointplot(x='temp', y='area', data=df, height=10)
plt.title('Joint distribution of temp and area')

#Show the plot
plt.show()


# # 4. Scatter Plot:

# In[24]:


#Create a scatter plot of temp, RH, DC, DMC variables
sns.pairplot(df[['temp', 'RH', 'DC', 'DMC']])

#Show the plot
plt.show()


# # 5. Open-ended Analysis:

# In[25]:


#Relative humidity distribution by month
#The violin plot shows the distribution of relative humidity by month, showing the variability in humidity levels month wise
plt.figure(figsize=(12, 6))
sns.violinplot(x=df['month'], y=df['RH'], data=df, order=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], palette='Set2')
plt.title("Relative Humidity Distribution by Month")
plt.xlabel("Month")
plt.ylabel("Relative Humidity (%)")
plt.xticks(rotation=45)
plt.show()

