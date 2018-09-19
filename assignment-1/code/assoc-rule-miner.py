import sys
import os


def read_csv(filepath):
    """
    Read transactions from csv_file specified by filepath
    Args:
        filepath (str): the path to the file to be read

    Returns:
        list: a list of lists, where each component list is a list of string representing a transaction

    """

    transactions = []
    with open (filepath, 'r') as f:
        lines = f.readlines ()
        for line in lines:
            transactions.append (line.strip ().split (',')[:-1])
    return transactions

def get_first_itemset(transactions):
    itemset = set ()
    for tnx in transactions:
        for item in tnx:
            itemset.add ([item])

def generate_frequent_itemset(transactions, itemset, minsup):
    """
    Generate the frequent itemsets from transactions
    Args:
        transactions (list): a list of lists, where each component list is a list of string representing a transaction
        minsup (float): specifies the minsup for mining

    Returns:
        list: a list of frequent itemsets and each itemset is represented as a list string

    Example: Output: [['margarine'], ['ready soups'], ['citrus fruit','semi-finished bread'], ['tropical fruit',
    'yogurt','coffee'], ['whole milk']] The meaning of the output is as follows: itemset {margarine}, {ready soups},
    {citrus fruit, semi-finished bread}, {tropical fruit, yogurt, coffee}, {whole milk} are all frequent itemset

    """

    return [[]]

def generate_association_rules(transactions, minsup, minconf):
    """
    Mine the association rules from transactions
    Args:
        transactions (list): a list of lists, where each component list is a list of string representing a transaction
        minsup (float): specifies the minsup for mining
        minconf (float): specifies the minconf for mining

    Returns:
        list: a list of association rule, each rule is represented as a list of string

    Example: Output: [['root vegetables', 'rolls/buns','=>', 'other vegetables'],['root vegetables', 'yogurt','=>',
    'other vegetables']] The meaning of the output is as follows: {root vegetables, rolls/buns} => {other vegetables}
    and {root vegetables, yogurt} => {other vegetables} are the two associated rules found by the algorithm


    """
    itemset = get_first_itemset(transactions)
    frequent_itemsets = generate_frequent_itemset(transactions, itemset, minsup)

    return [[]]


datafilepath = r'C:\Users\Sanjay Saha\CS5228-assignments\assignment-1\Association Rule Mining\Data\Groceries100.csv'
transactions = read_csv (datafilepath)
minsup = float (0.05)
minconf = float (0.3)
result = generate_association_rules (transactions, minsup, minconf)

# store associative rule found by your algorithm for automatic marking
with open ('.' + os.sep + 'Output' + os.sep + 'assoc-rule-result.txt', 'w') as f:
    for items in result:
        output_str = '{'
        for e in items:
            if e == '=>':
                output_str = output_str[:-1]
                output_str += '} => {'
            else:
                output_str += e
                output_str += ','

        output_str = output_str[:-1]
        output_str += '}\n'
        f.write (output_str)
