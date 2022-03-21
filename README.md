# customer_segmentation_analysis
--This is a case study that consists in identifying important factors for predicting whether a customer has converted or not.--

For that, I use the input data of the file 'DS_customer_data_sample.csv' that includes the following variables: </br>

| customer_id | Numeric id for a customer </br>
| converted | Whether a customer converted to the product (1) or not (0) </br>
| customer_segment | Numeric id of a customer segment the customer belongs to. It corresponds to catagorical data : 11, 12 and 13. </br>
| gender | Customer gender (male or female) </br>
| age | Customer age </br>
| related_customers | Numeric - number of people who are related to the customer </br>
| family_size | Numeric - size of family members </br>
| initial_fee_level | Initial services fee level the customer is enrolled to (numeric-float) </br>
| credit_account_id | Identifier (hash) for the customer credit account. If customer has none, they are shown as "9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0" </br>
| branch | Which branch the customer mainly is associated with </br>

To analyze the data, I provide two python files: (1) jupyter notebook 'customer_segmentation_analysis.ipynb' and (2) 'functions_for_analysis.py'. The latter contains functions required for the analysis.
