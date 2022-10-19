from webb import webb

webb.traceroute("www.google.com")
webb.traceroute("www.google.com", 'trace.txt')
print(webb.get_ip("www.google.com"))
