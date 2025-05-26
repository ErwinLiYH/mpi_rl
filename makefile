node1 = 244
node2 = 243
node3 = 239

setup:
	ssh -fN -L 9000:10.122.1.$(node1):9000 yl1118@login.isca.ex.ac.uk
	ssh -fN -L 9001:10.122.1.$(node1):9001 yl1118@login.isca.ex.ac.uk
	ssh -fN -L 9002:10.122.1.$(node1):9002 yl1118@login.isca.ex.ac.uk
	
	ssh -fN -L 9003:10.122.1.$(node2):9003 yl1118@login.isca.ex.ac.uk
	ssh -fN -L 9004:10.122.1.$(node2):9004 yl1118@login.isca.ex.ac.uk
	ssh -fN -L 9005:10.122.1.$(node2):9005 yl1118@login.isca.ex.ac.uk

	ssh -fN -L 9006:10.122.1.$(node3):9006 yl1118@login.isca.ex.ac.uk
	ssh -fN -L 9007:10.122.1.$(node3):9007 yl1118@login.isca.ex.ac.uk
	ssh -fN -L 9008:10.122.1.$(node3):9008 yl1118@login.isca.ex.ac.uk

search:
	ps aux | grep '[s]sh -fN -L' | awk '{for (i=11; i<=NF; i++) printf "%s%s", $$i, (i==NF ? ORS : OFS)}'

close:
	ps aux | grep '[s]sh -fN -L' | awk '{print $$2}' | xargs --no-run-if-empty kill