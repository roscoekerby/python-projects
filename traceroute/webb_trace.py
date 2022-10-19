import sys
# insert at position 1 in the path, as 0 is the path of this file.
sys.path.insert(1, './')
import webb_fixed

webb_fixed.traceroute("www.google.com")
webb_fixed.traceroute("www.google.com", 'trace.txt')
print(webb_fixed.get_ip("www.google.com"))
