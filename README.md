# customer_segmentation_analysis
--This is a case study that consists in identifying important factors for predicting whether a customer has converted or not.--

For that, I use the input data of the file 'DS_customer_data_sample.csv' that includes the following variables:

| customer_id | Numeric id for a customer
| converted | Whether a customer converted to the product (1) or not (0)
| customer_segment | Numeric id of a customer segment the customer belongs to. It corresponds to catagorical data : 11, 12 and 13.
| gender | Customer gender (male or female)
| age | Customer age
| related_customers | Numeric - number of people who are related to the customer
| family_size | Numeric - size of family members
| initial_fee_level | Initial services fee level the customer is enrolled to (numeric-float)
| credit_account_id | Identifier (hash) for the customer credit account. If customer has none, they are shown as "9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0"
| branch | Which branch the customer mainly is associated with 

The code is given in the jupyter notebook 'customer_segmentation_analysis.ipynb' and I provide another python file 'functions_for_analysis.py' that contains functions required for the analysis that I call thhrough thhe notebook.
