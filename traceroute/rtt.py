import time
import requests
import sys

def RTT(url):    
	url = sys.argv[1]
	start_time = time.time()
	
	request = requests.get(url)
	
	finish_time = time.time()
	
	elapsed_time = finish_time - start_time
	
	return elapsed_time
	
url = "http://www.google.com"
print("The Round Trip Time (RTT) to {} took {} seconds.".format(url, RTT(url)))
