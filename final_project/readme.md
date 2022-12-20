site : http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/ 

DATASET: High school contact and friendship networks
Release data: Jul 15, 2015

These data sets correspond to the contacts and friendship relations between students in a high school in Marseilles, France, in December 2013, as measured through several techniques.

-The first data set gives the contacts of the students of nine classes during 5 days in Dec. 2013, as measured by the SocioPatterns infrastructure. The file contains a tab-separated list representing the active contacts during 20-second intervals of the data collection. Each line has the form “t i j Ci Cj“, where i and j are the anonymous IDs of the persons in contact, Ci and Cj are their classes, and the interval during which this contact was active is [ t – 20s, t ]. If multiple contacts are active in a given interval, you will see multiple lines starting with the same value of t. Time is measured in seconds and expressed in UNIX ctime.

-The second data set corresponds to the directed network of contacts between students as reported in contact diaries collected at the end of the fourth day of the data collection. Each line has the form “i j w”, meaning that student i reported contacts with student j of aggregate durations of (i) at most 5 min if w = 1, (ii) between 5 and 15 min if w = 2, (iii) between 15 min and 1 h if w = 3, (iv) more than 1 h if w = 4.

-The third data set corresponds to the directed network of reported friendships. Each line has the form “i j”, meaning that student i reported a friendship with student j.

-The fourth data set corresponds to the list of pairs of students for which the presence or absence of a Facebook friendship is known. Each line has the form “i j w”, where w=1 means that students i and j are linked on Facebook, while w=0 means that they are not.

-Finally the metadata file contains a tab-separated list in which each line of the form “i Ci Gi” gives class Ci and gender Gi of the person having ID i.
Terms and conditions

The data are distributed to the public under a Creative Commons Attribution-NonCommercial-ShareAlike license. When this data is used in published research or for visualization purposes, please cite the following paper:

R. Mastrandrea, J. Fournet, A. Barrat,
Contact patterns in a high school: a comparison between data collected using wearable sensors, contact diaries and friendship surveys.
PLoS ONE 10(9): e0136497 (2015)