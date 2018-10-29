### Adequate
adequate.py is a tool for the generation of simple adequate subgraphs of small order that may appear in a stripped
contracted (A,B)-alternating graphs.
### Usage example

Here is a usage example of generations simple adequate subgraphs of order 6 that may appear in a stripped
contracted (A^3,B)-alternating graphs:

~~~
python adequate.py -l 6 -A 3 -B 1 \
	-m matchings/6.txt \
	-ad adequates/4.txt
~~~
* `-h` --help show this help message and exit
* `-l`        size of subgraphs
* `-A`        number dublications of genome A
* `-B`        is B or not (1 or 0)
* `-m`        path to file with l-size matchings 
* `-ad`       path to file with adequate graphs with size <l

### Output description
* `graphs.txt` all simple adequate graphs of order l. The graphs represent as adjacency matrices where A-edges have weight 1 and B-edges have weight 2.
* `graphs.tex` all simple adequate graphs of order l in the tex format. 
