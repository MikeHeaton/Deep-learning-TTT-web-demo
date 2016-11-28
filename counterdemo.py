# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:23:26 2016

@author: Mike
"""

from twisted.web import server, resource
from twisted.internet import reactor, endpoints

class Counter(resource.Resource):
    isLeaf = True
    numberRequests = 0

    def render_GET(self, request):
        self.numberRequests += 1
        request.setHeader(b"content-type", b"text/plain")
        request.setHeader(b"x-foo", b"bar")
        content = u"I am request #{}\n".format(self.numberRequests)
        return content.encode("ascii")

endpoints.serverFromString(reactor, "tcp:8080").listen(server.Site(Counter()))
reactor.run()

# So... we need to make the program generate HTML code, and send it to the
# website?
\   
