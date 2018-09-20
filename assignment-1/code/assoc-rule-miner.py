"""
Description     : Simple Python implementation of the Apriori Algorithm

Usage:
    $python assoc-rule-miner.py -f DATASET.csv -s minSupport  -c minConfidence

    $python assoc-rule-miner.py -f DATASET.csv -s 0.15 -c 0.6
"""

import sys
import os
from itertools import chain, combinations
from collections import defaultdict
from sys import exit

def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain (*[combinations (arr, i + 1) for i, a in enumerate (arr)])


def generate_freq_items(itemSet, transactionList, minSupport, freqSet):
    """calculates the support for items in the itemSet and returns a subset
       of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set ()
    localSet = defaultdict (int)

    for item in itemSet:
        for transaction in transactionList:
            if item.issubset (transaction):
                freqSet[item] += 1
                localSet[item] += 1

    for item, count in localSet.items ():
        support = float (count) / len (transactionList)

        if support >= minSupport:
            _itemSet.add (item)

    return _itemSet


def merge_set(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set ([i.union (j) for i in itemSet for j in itemSet if len (i.union (j)) == length])


def get_transaction_list(data_iterator):
    transactionList = list ()
    itemSet = set ()
    for record in data_iterator:
        transaction = frozenset (record)
        transactionList.append (transaction)
        for item in transaction:
            itemSet.add (frozenset ([item]))  # Generate 1-itemSets
    return itemSet, transactionList


def generate_association_rules(data_iter, minSupport, minConfidence):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    itemSet, transactionList = get_transaction_list (data_iter)

    freqSet = defaultdict (int)
    largeSet = dict ()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    assocRules = dict ()
    # Dictionary which stores Association Rules

    oneCSet = generate_freq_items (itemSet,
                                   transactionList,
                                   minSupport,
                                   freqSet)

    currentLSet = oneCSet
    k = 2
    while (currentLSet != set ([])):
        largeSet[k - 1] = currentLSet
        currentLSet = merge_set (currentLSet, k)
        currentCSet = generate_freq_items (currentLSet,
                                           transactionList,
                                           minSupport,
                                           freqSet)
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float (freqSet[item]) / len (transactionList)

    toRetItems = []
    for key, value in largeSet.items ():
        toRetItems.extend ([(tuple (item), getSupport (item))
                            for item in value])

    toRetRules = []
    for key, value in list (largeSet.items ())[1:]:
        for item in value:
            _subsets = map (frozenset, [x for x in subsets (item)])
            for element in _subsets:
                remain = item.difference (element)
                if len (remain) > 0:
                    confidence = getSupport (item) / getSupport (element)
                    if confidence >= minConfidence:
                        toRetRules.append (((tuple (element), tuple (remain)),
                                            confidence))
    return toRetItems, toRetRules


def read_csv(fname):
    file_iter = open (fname, 'r')
    for line in file_iter:
        line = line.strip ().rstrip (',')  # Remove trailing comma
        record = frozenset (line.split (','))
        yield record


def print_freq_items(outdir, result):
    with open (outdir + os.sep + 'Output' + os.sep + 'frequent_itemset_result.txt', 'w') as f:
        for items in result:
            items = items[0]
            output_str = '{'
            for e in items:
                output_str += e
                output_str += ','

            output_str = output_str[:-1]
            output_str += '}\n'
            f.write (output_str)


def print_rules(outdir, rules):
    with open (outdir + os.sep + 'Output' + os.sep + 'assoc-rule-result.txt', 'w') as f:
        for rule in rules:
            rule = rule[0]
            rule_left_side = rule[0]
            rule_right_side = rule[1]
            output_str = '{'
            for e in rule_left_side:
                output_str += e
                output_str += ','
            output_str = output_str[:-1]
            output_str += '} => {'
            for e in rule_right_side:
                output_str += e
                output_str += ','
            output_str = output_str[:-1]
            output_str += '}\n'
            f.write (output_str)


if __name__ == '__main__':

    datafilepath = r'C:\Users\Sanjay Saha\CS5228-assignments\assignment-1\Association Rule Mining\Data\Groceries100.csv'
    outdir = r'C:\Users\Sanjay Saha\CS5228-assignments\assignment-1\Association Rule Mining'

    if len (sys.argv) != 3 and len (sys.argv) != 4:
        print ("Wrong command format, please follwoing the command format below:")
        print ("python assoc-rule-miner-template.py csv_filepath minsup")
        print ("python assoc-rule-miner-template.py csv_filepath minsup minconf")
        exit (0)

    transactions = None
    minsup = 0.0
    minconf = 0.0
    if len (sys.argv) == 3:
        transactions = read_csv (sys.argv[1])
        minsup = float (sys.argv[2])
        minconf = 0.3
    elif len (sys.argv) == 4:
        transactions = read_csv (sys.argv[1])
        minsup = float (sys.argv[2])
        minconf = float (sys.argv[3])

    items, rules = generate_association_rules (transactions, minsup, minconf)

    if len (sys.argv) == 3:
        # Output frequent item-sets
        print_freq_items(outdir, items)
    elif len (sys.argv) == 4:
        # Output Rules to File
        print_rules(outdir, rules)

