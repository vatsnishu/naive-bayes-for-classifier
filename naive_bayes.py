#!/usr/bin/python

from csv import reader
from collections import Counter
from math import sqrt
from sklearn.model_selection import KFold
import math

fname = 'census-income.csv'

def loadData(fname):
	dataset=list()
	with open(fname, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def calculateTotal(tset):
	total=0
	for row in tset:
		total+=1
	return total

def summarize(tset):
	summaries={"mean":{},"mode":{},"stdev":{},"freq":{},"classTotal":0,"classProbability":0.0,"dataTotal":0}
	summaries["classTotal"]=calculateTotal(tset)
	for column in range(len(tset[0])-1):
		if continuous_or_nominal[column]==1:
			summaries["mean"][column]=calculate_mean(column,tset)
			summaries["stdev"][column]=calculate_stdev(column,tset,summaries["mean"][column])
		else:
			summaries["mode"][column]=calculate_mode(column,tset)
		summaries["freq"][column]=calculate_freq(column,tset)
	return summaries

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def calculate_mean(column,tset):
	number_of_non_missing_entries=0
	sum_of_non_missing_entries=0
	for row in tset:
		if is_number(row[column]):
			number_of_non_missing_entries+=1
			sum_of_non_missing_entries+=row[column]

	mean=sum_of_non_missing_entries/number_of_non_missing_entries
	return mean

def calculate_mode(column,tset):
	wordfreq={}
	for row in tset:
		if '?' in row[column]:
				pass
		else:
			if row[column] not in wordfreq:
				wordfreq[row[column]] = 0 
			else:
				wordfreq[row[column]] += 1
    	maximum = [(value, key) for key, value in wordfreq.items()]
	return max(maximum)[1]

def calculate_stdev(column,tset,mean):
	avg = mean
	number_of_entries=0
	variance=0.0
	for row in tset:
		variance+=pow(row[column]-avg,2)
		number_of_entries+=1
	variance = variance/number_of_entries
	return sqrt(variance)

def calculate_freq(column,tset):
	wordfreq={}
	for row in tset:
		if row[column] not in wordfreq:
			wordfreq[row[column]] = 0 
		else:
			wordfreq[row[column]] += 1
    	maximum = [(value, key) for key, value in wordfreq.items()]
	return wordfreq

def separateByClass(tset):
	separated = {}
	for row in tset:
		if (row[-1] not in separated):
			separated[row[-1]] = []
		separated[row[-1]].append(row)
	return separated

def summarizeByClass(tset):
	separated = separateByClass(tset)
	summariesByClass = {}
	total=calculateTotal(tset)
	for classValue, instances in separated.iteritems():
		summariesByClass[classValue] =summarize(instances)
		summariesByClass [classValue]["dataTotal"]=total
		summariesByClass[classValue]["classProbability"]=float(summariesByClass[classValue]["classTotal"])/float(summariesByClass[classValue]["dataTotal"])
	return summariesByClass

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	res=float(1 / (math.sqrt(2*math.pi) * stdev) * exponent)
	if(res>0.0):
		return math.log(res)
	else:
		return 0.0

def calculateProbability_nom(freq, classTotal):
	if(freq>0):
		return math.log(float(freq)/float(classTotal))
	else:
		return 0.0
	

def calculateClassProbabilities(summaries, tseti):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for column in range(len(tseti)-1):
			if column==24:
				continue
			if continuous_or_nominal[column]==1:
				mean=classSummaries["mean"][column]
				stdev=classSummaries["stdev"][column]
				x = tseti[column]
#				probabilities[classValue] += calculateProbability(x, mean, stdev)
			else:
				x = tseti[column]
				if(x in (classSummaries["freq"][column])):
					freq= classSummaries["freq"][column][x]
					classTotal=classSummaries["classTotal"]
					probabilities[classValue] += calculateProbability_nom(freq, classTotal)
					
				else:
					continue
		probabilities[classValue] += math.log(classSummaries["classProbability"])	
	return probabilities

def predict(summaries, tseti):
	probabilities = calculateClassProbabilities(summaries, tseti)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, tSet):
	predictions = []
	for i in range(len(tSet)):
		result = predict(summaries, tSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

###################
###################
##     MAIN      ##
###################
###################

dataset=loadData(fname)

#Here 1 suggests continuous data and 0 suggests nominal(categorial) data in column
continuous_or_nominal={0:1,1:0,2:0,3:0,4:0,5:1,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:1,17:1,18:1,19:0,20:0,21:0,22:0,23:0,24:1,25:0,26:0,27:0,28:0,29:0,30:1,31:0,32:0,33:0,34:0,35:0,36:0,37:0,38:0,39:1,40:0}

#Summary of whole data
FullDatasetSummary={"mean":{},"mode":{},"stdev":{},"freq":{},"total":0, "numberofEmptyEntries":{}}

############################################################################
# calculation of Full data summary and modifying empty fields with mean in # 
# continuous data columns and mode in nominal(categorical) data column     #
############################################################################

print "column_no              mean/mode                   stdev              Number of Empty Entries"
print "_____________________________________________________________________________________________"

FullDatasetSummary["total"]=calculateTotal(dataset)
for column in range(len(dataset[0])-1):
	tempEmptyEntries=0
	if continuous_or_nominal[column]==1:  #Continuous data
		for row in dataset:
			if is_number(row[column]):    
				row[column]=float(row[column])  #non empty entry made float
		FullDatasetSummary["mean"][column]=calculate_mean(column,dataset)
		#Filling with mean in empty entries
		for row in dataset:
			#print row[column]
			float(row[column])
			if is_number(row[column]):
				pass
			else:
				tempEmptyEntries+=1
				row[column]=FullDatasetSummary["mean"][column]				
		FullDatasetSummary["stdev"][column]=calculate_stdev(column,dataset,FullDatasetSummary["mean"][column])
		FullDatasetSummary["numberofEmptyEntries"][column]=tempEmptyEntries
		print str(column)+"\t\t\t"+str(FullDatasetSummary["mean"][column])+"\t\t\t"+str(FullDatasetSummary["stdev"][column])+"\t\t\t"+str(FullDatasetSummary["numberofEmptyEntries"][column])

	else:									# nominal(categorical) data
		FullDatasetSummary["mode"][column]=calculate_mode(column,dataset)
		#Filling with mode in empty entries
		for row in dataset:
			if '?' in row[column]:
				tempEmptyEntries+=1
				row[column]=FullDatasetSummary["mode"][column]
			else:
				pass
		
		FullDatasetSummary["numberofEmptyEntries"][column]=tempEmptyEntries
		print str(column)+"\t\t\t"+str(FullDatasetSummary["mode"][column])+"\t\t\tNA\t\t\t"+str(FullDatasetSummary["numberofEmptyEntries"][column])

		FullDatasetSummary["freq"][column]=calculate_freq(column,dataset)

#separate By classs
separated = {}
for row in dataset:
	if (row[-1] not in separated):
		separated[row[-1]] = []
	separated[row[-1]].append(row)

print 
print "Number of classes= "+str(len(separated))

total=0
for i in separated:
	print "Class "+str(i)+" has "+str(len(separated[i]))+" items"
	total+=len(separated[i])
	print
print "Database "+str(i)+" has "+str(total)+" items"
print
for column in range(len(dataset[0])-1):
	if continuous_or_nominal[column]==1:  #Continuous data
		pass
	else:
		print column
		print FullDatasetSummary["freq"][column]




###########################################################################
# Calculating the accuarcy by running naive bayes of train data and       # 
# testing on test data                                                    #
###########################################################################
print
print "epoch                    Accuracy_MEAN              Accuracy_STANDARD_DEVIATION"
print "_______________________________________________________________________________"
n_epochs=30
j=0
for i in range(n_epochs):
	j=j+1
	accuracies=[]
	folds = KFold(n_splits=10, shuffle=True)
	for train_set , test_set in folds.split(dataset):
		x_tr=[]
		x_test=[]
		for i in test_set:
			x_test.append(dataset[i])
		for i in train_set:
			x_tr.append(dataset[i])
		summariesByClass = summarizeByClass(x_tr)
		predictions = getPredictions(summariesByClass, x_test)
		accuracy = getAccuracy(x_test, predictions)
		accuracies.append(accuracy)

	number_of_accuracies=0
	sum_of_accuracies=0
	for acc in accuracies:
		number_of_accuracies+=1
		sum_of_accuracies+=acc
	accuracies_mean=sum_of_accuracies/number_of_accuracies

	avg =accuracies_mean
	number_of_accuracies=0
	accuracies_variance=0.0
	for acc in accuracies:
		accuracies_variance+=pow(acc-avg,2)
		number_of_accuracies+=1
	accuracies_variance = accuracies_variance/number_of_accuracies
	accuracies_variance=sqrt(accuracies_variance)
	print " "+str(j)+"                    "+str(accuracies_mean)+"                     "+str(accuracies_variance)



