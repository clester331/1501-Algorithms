# 1501 Algorithms - Basically the canvas study guide but now I don't have to go through all the slides anymore

## Compression Algorithms

### LZW 

#### Compression
* Initialize codebook to all single characters
  * Character maps to its ASCII value
  * Codewords 0 to 255 are filled now
* While !EOF (End Of File)
  * Find the longest match in the codebook
  * Output codeword of that match
  * Take this longest match + the next character in the file
    * Add that to the codebook with the next available codeword value
  * Start from the character right after the match
  
* In general, LZW will give better compression than Huffman
  * Very long patterns can be built up, leading to better compression
  * Different files don't hurt each other as they did in Huffman
  * LZW is also better for compressing archived directories of files
  
* Compress using 12-bit codewords
  * TOBEORNOTTOBEORTOBERNOT
  
| Cur | Output | Add    |
|:---:|:------:|:---:   |
|T    |84      |TO:256  |
|O    |79      |OB:257  |
|B    |66      |BE:258  |
|E    |69      |EO:259  |
|O    |79      |OR:260  |
|R    |82      |RN:261  |
|N    |78      |NO:262  |
|O    |79      |OT:263  |
|T    |84      |TT:264  |
|TO   |256     |TOB:265 |
|BE   |258     |BEO:266 |
|OR   |260     |ORT:267 |
|TOB  |265     |TOBE:268|
|EO   |259     |EOR:269 |
|RN   |261     |RNO:270 |
|OT   |263     |------- |

#### LZW Expansion
* Initialize codebook to all single characters
  * ASCII value maps to its character
* While !EOF
  * Read next codeword from file
  * Lookup corresponding pattern in the codebook
  * Output that pattern
  * Add the previous pattern + the first character of the current pattern to the codebook
  
| Cur | Output | Add    |
|:---:|:------:|:---:   |
|84   |T       |--------|
|79   |O       |256:TO  |
|66   |B       |257:OB  |
|69   |E       |258:BE  |
|79   |O       |259:EO  |
|82   |R       |260:OR  |
|78   |N       |261:RN  |
|79   |O       |262:NO  |
|84   |T       |263:OT  |
|256  |TO      |264:TT  |
|258  |BE      |265:TOB |
|260  |OR      |266:BEO |
|265  |TOB     |267:ORT |
|259  |EO      |268:TOBE|
|261  |RN      |269:EOR |
|263  |OT      |270:RNO |

#### Expansion/Corner Case
* Expansion can sometimes be a step ahead of compression
  * If during compression, the (pattern, codeword) that was **just added** to the directory is **immediately used** in the next step, the decompression algorithm will not yet know the codeword
  * This is easily detected and dealt with however (Use previous output + 1st character of the previous output)
  
#### Adaptive LZW/Variable Width Codewords
* How long should codewords be?
  * Use fewer bites
    * Gives better compression earlier on
    * But leaves fewer codewords available, which will hamper compression later
  * Use more bits:
    * Delays actual compression until longer patterns are found due to large codeword size
    * More codewords available means that greater compression gains can be made later on in the process
    
* How variable width works
  * Start out using 9 bit codewords
  * When codeword 512 (n-bits<sup>2</sup>) is inserted into the codebook, switch to outputting/grabbing 10-bit codewords
  * When codeword 1024 is inserted into the codebook, switch to outputting/grabbing 11 bit codewords
  * **When you reach n<sup>2</sup> codewords (n = # of bits),  then start outputting/grabbing n+1-bit codewords**
  
* What to do when out of codewords
  1. Stop adding new keywords and use the codebook as it stands
      * Maintains long already established patterns
      * But if the file changes, it will not be compressed as effictively
  2. Throw out the codebook and start over from single characters
      * Allows new patterns to be compressed
      * Until new patterns are build up though, compression will be minimal
    
### Shannon's Entropy and Limits on Compression
* **Shannon Information** is a measure of the **unpredictabillity of information content**
* Shannon Information of a message m
  * I(m) = -1 * log<sub>2</sub>Pr(m) bits
    * Pr(m) is the probability of message m
  * Examples:
    * Pr(c1) = 0.5 -> I(c1) = -1 * log<sub>2</sub>(0.5) = -1 * -1 = **1 bit**
    * Pr(c2) = 0.25 -> I(c2) = -1 * log<sub>2</sub>(0.25) = -1 * -2 = **2 bits**
    * Pr(c3) = 1/2<sup>100</sup> -> I(c3) = -1 * log<sub>2</sub>(2<sup>-100</sup>) = -1 * -100 = **100 bits**
    
* **Shannon's Entropy** is a keay measure in information theory
* Entropy of an information source (like a file)
  * H = sum <sub>all unique messages m</sub> Pr(m) * I(m)
  * Average of Shannon's Information of all unique messages
* Entropy per bit = H / file size in bits

* How can we determine the probability of each character in the file?
  * Pr(c) = f(c) / file size (f = frequency) <sub>I think?</sub>
  
* By losslessly compressing data, we represent the same information in less space
  * Entropy of original file = entropy of compressed file
* On average, a loseless compression scheme caannot compress a message to have more than 1 bit of entropy per compressed message

* **Entropy of a language**: The average number of bits required to store a letter of the language
  * Uncompreesed English has between 0.6 and 1.3 bits of entropy per letter
    * Entropy of a language * length of messagein chracters = amount of information contained in that message
    
### Burrows-Wheeler
* Best compression algorithm (in terms of compression ratio) for text
* **Three steps of Burrows-Wheeler**
  1. Burrows-Wheeler Transform
      * Cluster same letters as close to each other as possible
  2. Move-To-Front Encoding
      * Convert output of precious set into an integer file with large **frequency** differences
  3. Huffman Compression
      * Compress the ile of integers using Huffman Compression
* **For expansion, apply the inverse of compression steps in reverse order (3, 2, 1)

#### Move-To-Front Encoding
* Initaialize an ordered list of 256 ASCII characters
  * character *i* appears *i*th in the list
* For each character c from input
  * Output the index in the list where c appears
  * move c to the front of the list (i.e., index 0)

|*Input* | e | a | e | d | e | e |
|:------:|:-:|:-:|:-:|:-:|:-:|:-:|
|*0 - a* | e | a | e | d | e | e |
|*1 - b* | a | e | a | e | d | d |
|*2 - c* | b | b | b | a | a | a |
|*3 - d* | c | c | c | b | b | b |
|*4 - e* | d | d | d | c | c | c |
|*Output*|**4**|**1**|**1**|**4**|**1**|**0**|

*In the output of MTF Encoding, smaller integers have higher frequencies than larger integers

##### Move-To-Front Decoding
* Same as encoding, except start from number and go to letters

#### Burrows Wheeler Transform
* Rearranges the characters in the input
  * Lots of clusters with repeated chracters
  * Still possible to recover original input
  
* **Pseudocode**:
  * For each block of length N characters
    * Generate **N strings** by **cycling** the characters of the block one step of the time
    * Sort the strings
    * Output the last column in the sorted table and the index of the original block in the sorted array

* Example: "ABRACADABRA" (N = 11)

![image](https://user-images.githubusercontent.com/122314614/233750127-34751eed-d8ce-45dd-a37c-6a366c7cd3cb.png)

#### Burrows-Wheeler Transform Decoding
*How to recover original string? (ABABABA)
  1. Sort The encoded string (BBBAAAA -> AAABBB)
  2. Fill an array next[]
      * Defined for each entry in the sorted array
      * Holds the index in sorted array of the next string in the original array
      * Scan through the first column
        * For each row *i* holding character *c*
        * next[i] = first unassinged index *c* in the last column
  3. Recover the input string using next[] array
      
##### Downsides of Burrows-Wheeler
* process blocks of input file
  * Compared to LZW which processes the input one character at a time
* The larger the block size, the better the compression

## Graph Algorithms

### Intro to Graphs
* A graph G = (V, E)
  * Where V is a set of vertices
  * E is a set of edges connecting vertex pairs
* Example
  * V = {0, 1, 2, 3, 4, 5}
  * E = {(0, 1), (0, 4), (1, 2), (1, 4), (2, 3), (3, 4), (3, 5)}
  
![image](https://user-images.githubusercontent.com/122314614/233757467-4ae17eae-c3d7-4c80-a905-62201ab30aaa.png)

#### Adjacency Lists
* Array of Neightbor Lists
  * A[i] contains a list of the neighbors of vertex i
  ![image](https://user-images.githubusercontent.com/122314614/233757498-426268c3-559e-4bd7-b1c2-03d5993a53a4.png)
* Runtime
  * Check if two vertices are neighbors
  * Find the list of neighbors of a vertex
    * Θ(d)
      * d is the degree of a vertex (# of neighbors)
    * O(V)
  * Space?
    * Θ(v + e) memory

#### Graph Definitions
* Undirected Graph
  * Edges are unordered Pairs: (A, B) == (B, A)
  * Maximum Value = V*(V-1)/2
* Directed Graph
  * Edges are ordered pairs: (A, B) != (B, A)
  * Maximum Value = V<sup>2</sup>
* Adjacent Vertices, or Neighbots
  * Vertices connected by an edge
* Sparce Graph
  * e <= v lg v
* Dense Graph
  * e == MAX - ε
* Complete Graph
  * Has the maximum Number of edges. there is an edge between A and B for each pair of vertices A and B
* Connected Graph
  * there is a path from A to B for each pair of vertices A and B
  
#### Adjacency Matrix
* Rows/columns are vertex labels
  * M[i][j] = 1 if i and j are neighbors (0 if not neighbors)
  ![image](https://user-images.githubusercontent.com/122314614/233757846-c224e1a1-0204-4a96-97e2-81945d7e8fa0.png)
* Runtime
  * Check if two vertices are neighbors
    * Θ(1)
  * Find the list of neighbors of a vertex
    * O(V)
* Spaces
  O(v<sup>2</sup>
  
### Simple Graph Traversals (DFS, BFS) 
* BFS (Breadth-first search)
  * Search all directions evenly
    * from i, visit all of i's neighbors, then all of their neighbors, and so on
  * Would help us compute the distance between two vertices
* DFS (Depth-first search)
  * "Dive" as deep as possible into the graph first
  * Branch when necessary
  
### BFS
* BFS Traversals can be further used to determine the shortest path between two vertices
* Can be easily implemented using a queue
  * For each vertex listed, add all of its neighbors to the Q (if not previously added)
    * Vertices that have been seen (added to the Q) but not yet visited are said to be **fringe**
  * Pop head of the queue to be the next visited vertex![image](https://user-images.githubusercontent.com/122314614/233758107-12012005-ae5e-4b6b-96a9-f2f3624291ee.png)
  
* Pseudo-Code
```java
Q = new Queue
BFS(vertex v) { 
  add v to Q
  while(Q is not empty) {
    w = remove head of Q
    visited[w] = true //mark w as visited
    for each unseen neighbor x {
      seen[x] = true //mark x as seen
      parent[x] = w
      add x to Q
    }
  }
}
```
### Finding Connected Components For BFS
* A connected component is a connected sub-graph H'
  * (V', E')
    *  V' ⊆ V (Subset - every element of V' is in V)
    *  E’ = {(u, v) ∈ E and both u and v ∈ V’}
* To find all connected components
  * Wrapper function around BFS
  * A loop in the wrapper function wil have to coninually call bfs() while **there are still unseen vertices**
  * Each call will yield a spanning treee for a connected component of the graph
 
#### BFS Pseudocode For Connected Components
 ```java
int components = 0
for each vertex v in V {
  if visited[v] = false {
    components++
    Q = new Queue
    BFS(v)
  }
}

BFS(vertex v) {
  add v to Q
  component
  while(Q is not empty) {
    w = remove head of Q
    visited[w] = true
    component[w] = components
    for each unseen neighbor x {
      seen[x] = true
      add x to Q
    }
  }
}
```

#### BFS runtime
* Total time: **vertex processing time + edge processing time**
* each vertex is added to the queue exactly once and removed exactly once
  * v add/remove operations
    * O(v) time for vertex processing
* Edges are processed when adding the list of neighbors to the queue
* For adjacency lists: Total time is also **vertex processing time + edge processing time** (O(e) for edge processing)
  * O(v + e) 
* For adjacency matrix: O(v<sup>2</sup> time for edge processing
  * O(v<sup>2</sup) + v) = O(v<sup>2</sup>
* Running time depends on data structure selection

### DFS
* Used for Huffman Encoding
* Can be implemented recursively
  * For each vertex, visit first unseen neighbor
  * Backtrack at deadends (vertices with no unseen neighbors)
    * Try next unseen neighbor after backtracking
  * Arbitrary order of neighbors is assumed

#### DFS Pseudocode
```java
DFS(vertex v) {
  seen[v] = true //mark v as seen
  visit v //pre-order DFS
  for each unseen neighbor w {
    parent[w] = v
    DFS(w)
    //(re)visit v //in=order DFS
  }
  // visit v //post-order DFS
}
```

![image](https://user-images.githubusercontent.com/122314614/233795116-5d302659-2fa1-429e-b01a-11f43521bff3.png)

#### DFS Runtime
* For adjacency lists: Total time = vertex processing time + edge processing timre
* Each vertex is seen then visited exactly once
  * O(v) time for vertex processing (except for in order DFS - VTP is included in edge processing in this case)
* Edges are processed when finding the list of neighbors
* Each edge is checked at most twice, one per edge endpoint (O(e) for EPT)
* **Total time = O(v + e)

#### Runtime Comparisons for BFS and DFS
* At a high level, DFS and BFS both have the same runtime
  * O(v + e) for adjacency lists
  * O(v<sup>2</sup> for adjacency matrix
    * For dense graphs, v + e = O(v<sup>2</sup>)

### Articulation Points and Biconnected Graphs
* A *biconnected graoh* has at least 2 distinct paths between all vertex pairs
  * A distinct path shares no common edges or vertices with another path except for the start and end vertices
* A graph is biconnected iff it has **zero articulation points**
  * Articulation points are vertices, that if removed, will seperate the graph

#### Finding articulation points
* A DFS traversal builds a spanning tree (red edges)
* Edges not included in the spanning tree are called back edges [(4, 9) and (2, 6) in the picture)

![image](https://user-images.githubusercontent.com/122314614/233795582-d063ae79-0806-46b4-9425-63d2686ac917.png)

* Pre order DFS traversal visits the vertices in some order
  * use num(v) to number the vertices with their traversal order
  ![image](https://user-images.githubusercontent.com/122314614/233795661-bbd18824-d567-4400-9ec6-35436c101cc0.png)

* For each non-root vertex v, find the lowest numbered vertex reaachable from v
  * **not through v's parent, but by using 0 more more tree edges then at most one back edge**
* Move down the tree looking for a back edge that goes backwards the furthest
* In the image above, the articulation points are 4, 1, 2, and 0

#### low(v)
* Low(v) = The minimum of
  * num(v)
  * num(w) for all back edtges (v, w)
  * low(w) of all children of v
* Low(v) = lowest-numbered vertex reachable from v using 0 more more spanning tree edges and then at most one back edge

#### Articulation Point Pseudocode
```java
* A DFS visits each vertex v
  * label v with the two numbers
    * num(v)
    * low(v): initial value is num(v)
  * for each neighbor w
    * if already seen -> we have a back edge
      * update low(v) to num(w) if num(w) is less
    * if not seen -> we have a child
      * call DFS on the child
      * after the call returns,
        * Update low(v) to low(w) if low(w) is less
```
* num(v) is computed as we move down the tree (pre order)
* lowv) is updated as we move down and up the tree
* Recursive DFS is convenient to compute both

#### Finding articulation points of an undirected graph
```java
int num = 0
DFS(vertex v) {
  num[v] = num++
  low[v] = num[v] //initially
  seen[v] = true //mark v as seen
  for each neighbor w {
    if(w unseen) {
      parent[w] = v
      DFS(w) //after the call returns low[w] is computed, why?
      low[v] = min(low[v], low[w])
      if(low[w] >= num[v]) v is an articulation point
    } 
    else { //seen neighbor
      if(w != parent[v]) //and not the parent, so back edge {
        low[v] = min(low[v], num[w])
      }
    }
  }
}
```

### Prim's MST Algorithm
* MSTs are spanning trees that have the **minimum sum** of the wieights of its edges
* Prims Algorithm:
  * Initialize T to contain the starting vertex
    * T will eventually become the MST
  * While there are vertices not in T:
    * Find a **minimum edge-weight edge** that connects a vertex **in T** to a vertex **not yet in T**
    * Add the edge with its vertex to T   

![image](https://user-images.githubusercontent.com/122314614/233799072-5550ee36-5511-4c62-b32a-943cea241b00.png)

#### Prim's Runtime
* At each step, check all possible edges
* Runtime evaluates to Θ(v<sup>3</sup>)
* For Prim's **we do not have to look through all remaining edges**. Just the best edge for each possible vertex
* The best edge of a vertex is the edge with the **minimum weight** connecting the vertex from a vertex already 
  * Best edge values can be updated as we add vertices to T

#### Enchanced version of Prims
* Add start vertex to T
* Repeat until all vertices aded to T
  * check the neighbors of the added vertex
    * update best edge values if needed
    * Update parent as well
  * Add to T a vertex with the smallest best edge
![image](https://user-images.githubusercontent.com/122314614/233799332-36463d15-b8d8-4468-ab46-922f25516700.png)
* Runtime:
  * Update parent/best edge arrays: Θ(1)
  * Time to pick next vertex: Θ(v)
  * **Total time: Θ(v<sup>2</sup>)**

#### Prim's MST Algorithm Pseudocode
```java
seen, parent, and BestEdge are arrays of size v
Initialize seen to false, parent to -1, and BestEdge to infinity
BestEdge[start] = 0
for i = 0 to v-1 {
  Find w.s.t.seen[w] = false and BestEdge[w] is minimum over all unseen vertices
  seen[w] = 1
  for each neighbor x of w {
    if(BestEdge[x] > edge weight of edge (w, x) {
      BestEdge[x] = edge weight of (w, x) {
      parent[x] = w
    }
  }
}
```
* **The parent array represents the found MST**

#### Eager Prim's
* PQ will need to be indexable to update the best edge
* Runtimes:
  * v inserts: v log v
  * e updates: e log v
  * v removeMin: v log v
  * **Total runtime = e log v**

#### Lazy Prims
* PQ is not indexable for lazy prims
* Implementation:
  * Visit a vertex
  * Add edges coming out of it to a PQ
  * while there are unvisited vertice:
    * pop from the PQ for the next vertex to visit and repeat 
![image](https://user-images.githubusercontent.com/122314614/233799601-cb7efede-d129-461e-8402-e7b570c7c329.png)
* Must insert all e edges into the priority queue
  * In the worst case we'll also have to remove all e edges
* Runtime: Θ(e lg e)

#### Comparison of Prim's Implementations
* Parent/Best edge array Prims:
  * Runtime: Θ(v<sup>2</sup>)
  * Space: Θ(v)
* Lazy Prim's:
  * Runtime: Θ(e lg e)
  * Space: Θ(v)
  * Requires a PQ
* Eager Prims:
  * Runtime: Θ( e lg v)
  * Space Θ(v)
  * requires an indexable PQ  

### Kruskal's MST Algorithm
* Insert all edges into a PQ
* Grab the minimum edge from the PQ that does not create a cycle in the MST
* Remove it from the PQ and add it to the MST
![image](https://user-images.githubusercontent.com/122314614/233799872-b312feef-f570-4c4b-a222-ac4f2d6e3cfa.png)

#### Kruskals Pseudocode
* Insert all *e* edges into a PQ
* T = an empty set of edges
* Repeat until T contains v-1 edges
  * Remove a min edge from the PQ
  * Add the edge to T if the edge does not create a cycle in T
* return T  

#### Kruskal's Runtime
* Instead of building up the MST starting from a single vertex, we build it up using edges all over the graph
* How to implement cycle detection
  * BFS/DFS
    * v + e
  * Union/find data Structure (NOT COVERED)
    * log v
* **Total Runtime: Θ(e log v)** (same as Prim's

### Dijkstra's Algoritm
* distance[]: Best known shortest distance from start to each vertex
* distance[start] = 0
* distance[x] = Double.POSITIVE_INFINITY for other vertices
* Dijkstra's is only correct when all edge weights >= 0

#### Dijsktra's Pseudocode
* cur = start
* While destination not visited 
  * For each unvisited neighbor x of cur 
    * Compute shortest distance from start to x through cur
      * = distance[cur] + weight of edge from cur to x
    * Update distance[x] if distance through cur < distance[x]
  * Mark cur as visited
  * cur = an unvisited vertex with the smallest distance
![image](https://user-images.githubusercontent.com/122314614/233805102-79927838-d0e3-43de-aa3c-523ae8fdf528.png)
* The distance array keeps track of the best path from the start
  * Compare to best edge array in Prim's
* Once a vertex is visited, its distance value **doesn't change**
* Parent array used to construct shortest path

#### Dijsktra's Runtime
* Depends on implementation
  * Distance and Parent arrays: Θ(v<sup>2</sup>
  * Priority Queue: Θ(e log v)
* Algorithm may stop earlier when destination visited

#### Bi-Directional search
* Start two instances of Dijkstra's possibly in parallel
  * From source on original graph
  * From destination, on reverse graph
* When processing an edge to a vertex visited by the other instance, update shortest known distance between start and destination
* Stop when tops of both heaps give a distance >= sortest known  

#### A* Search
* Use lower bound estimates for the distance of the rest of the path to destination
  * Pick vertex with minimum distance[v] + estimate[v]
* Lower bound estimates using landmarks and triangual inequality
* Requires preprocessing to compute and store distnaces from each vertex to each landmark

### Bellman-Ford's Algorithm (and negative edge weights)
* distance[v] = Double.POSITIVE_INFINITY (for all vertices except start)
* distance[start] = 0
* Runtime: O(v * e)

#### Bellman-Ford Pseudocode
``` java
repeat v-1 times {
  for each vertex cur {
    for each neighbor x of cur {
      compute shortest distance from start to x via cur
        = distance[cur] + weight of cur(cur, x)
       if computed distance < distance[x]
         Update distance[x] and parent[x]
    }
  }
}
``` 

#### BMF Optimization
* initialize a FIFO (first in, first out) Queue Q. 
  * Pop a vertex *cur* from Q and if computed distance < distance[x], then add x to Q if it isn't already there 
![image](https://user-images.githubusercontent.com/122314614/233805917-9a46c3e0-3b2c-44d0-b699-e6e2ad084ade.png)
* Bellman-Ford's is correct event when there are negative edge weights in the graph

#### Negative Cycles
* BMF won't terminate if a negative cycle exists
* Finding a negativ cycle:
  * For each vertex cur:
    * For each neighbor x of cur
      * Compute shortest distance from start to x via cur
        = distance[xur] + weight of (cur, x)    
      * if computed distance < distance[x]
        * Update distance[x] and parent[x]
* If another iteration results in update of distance[v] for a vertex v, then v is in a negaitve cycle
* To detect a negative cycle reachable from the start:
  * Build a graph using parent to child links set by BMF
  * Modify DFS to detect if a cycle exists
    * If a neighbor is already visited and is on the runtime stack
      * We have a cycle
      * follow parent links until back to current node
      * Add up edge weights
      * If negative, stop; otherwise continue    

## Priority Queues

### ADT Priority Queue
* Primary Operations
  * Insert
  * Find item with highest priority
    * findMin() / findMax()
  * Remove an item with highest priority
    * removeMix() / removeMax()

#### Implementations
|Type          |findMin|removeMin|insert |
|:------------:|:-----:|:-------:|:-----:|
|Unsorted Array|O(n)   |O(n)     |O(1)   |
|Sorted Array  |O(1)   |O(1)     |O(n)   |
|Red-Black BST |O(logn)|O(logn)  |O(logn)|

* Armortized Runtime = Total runtime of a sequence of operations / # of operations


### The Heap and Implementations
* A heap is a complete binary tree such that for each node T in the tree:
  * T.item is of a higher priority than T.right_child.item
  * T.item is of a higher priority than T.left_child.item
* **This is the heap property**
* It does not matter how T.left_child relates to T.right_child.item
* In a Min Heap, a highest priority itme is a **minimum in the tree** (the root of the tree)
![image](https://user-images.githubusercontent.com/122314614/233797719-4bfe0cf9-7a08-42e4-ae30-60bc091f2d13.png)

#### Heap PQ Runtimes
* find: the root of the tree, so Θ(1)
* Insert:
  * Add the last inserted itme at the next available leaft (left to right)
  * Push the new item up the tree until heap property established
* Remove
  * Overwrite the root with the item from the last leaf
    * Then delete the last leaf
  * The new root may violate the heap property
    * Push the new root down the tree until heap propery established
 * **Insert and remove are both Θ(log n)**
 * Updated ADT PQ implementations with Heap:

|Type          |findMin|removeMin|insert |
|:------------:|:-----:|:-------:|:-----:|
|Unsorted Array|O(n)   |O(n)     |O(1)   |
|Sorted Array  |O(1)   |O(1)     |O(n)   |
|Red-Black BST |O(logn)|O(logn)  |O(logn)|
|Heap          |O(1)   |O(logn)  |O(logn)|

#### Heap Implementations 
* Linked: tree nodes like for BinaryTree
  * overhead for dynamic node allocation
  * Must have parent links
* Array: a heap is a complete binary tree
  * Can easily represent a complete binary tree using an array

#### Heapify Algorithm
* Scan through the array **right to left** starting from **the rightmost non-leaf**
  * Push item down the tree until heap property established
* Rightmost non-lead is at the largest index *i* such that leaf_child(i) < n
  * 2n+1 < n
  * i = floor((n-1)/2) if n is even (= (n-1)/2 - 1 if n is odd)
![image](https://user-images.githubusercontent.com/122314614/233798209-db196142-ac2e-469e-be75-243ea9c5edae.png)
* Runtime:
  * Upper Bound: O(n log n) 
  * Tighter analysis: for each node, we ake at most **height[node]** comparisons/swaps (= number of edges to deepest leaf)
    * So O(largerst term) or O(n)
 
#### Heap Sort
* Heapify the numbers
  * MAX heap to sort ascending and MIN heap to sort descending
* "Remove" the root (don't delete leaf node)
* Repeat   
* Worst Case Runtime: O(n log n)

### Indexible PQ implementation
* In order to update a new item in the heap, we can create a new ADT to be an ADT indexible PQ
* Runtime to find an arbitrary item a heap
  * Θ(n) (same for updating an item in the heap)
* We can also maintain a second data structure that maps item IDs to each item's current position in the heap
  * This creates an indexable PQ (indirection)

#### Example Indexable PQ pseudocode
```java
class CardPrice implements Comparable<CardPrice>{
  public String store;
  public double price;
  public CardPrice(String s, double p) { … }
  public int compareTo(CardPrice o) {
    if (price < o.price) { return -1; }
    else if (price > o.price) { return 1; }
    else { return 0; }
  }
}
```

## Memoization And Dynamic Programming Algorithms

### Fibonacci
* To account for all subproblems in the fibonacci program, we can use two different solutions to avoid running into the same subproblem
* Memoization Solution:
  * efficient: solves each subproblem only once
  * Still recursive
* Dynamic Programming:
  * Efficient: Also solves each subproblem only once
  * Iteratice
  * Allows for space optimization

#### Fibonacci Pseudocodes
```java
int[] F = new int[n+1];
F[0] = 0;
F[1] = 1;
for(int i = 2; i <= n; i++) { F[i] = -1 };

int fib_mem(n) {
  if (F[n] == -1) {
    F[n] = fib_mem(n-1) + fib_mem(n-2);
  }
  return F[n];
}
```
* Bottom-up approach
```java
int[] F = new int[n+1];
F[0] = 0;
F[1] = 1;

int bottomup_fib(n) {
  for(int i = 2; i <= n; i++) {
    F[i] = F[i-1] + F[i-2];
  } 
  return F[n];
}
```
* Improved Botton-up approach
```java
int improve_bottomup_fib(n) {
  if (n == 0) return 0;
  if (n == 1) return 1;
  int prev = 0; int cur = 1;
  for (int i = 0; i < n; i++){
    int new = prev + cur;
    prev = cur;
    cur = new;
  }
  return cur;
}
```

### Dynamic Programming
* Avoid solving the same sub-problem twice
* Iterative:
  * Start with smaller subproblems then larger subproblems
* Sometimes possible to optimize space needed  

#### DP Steps
* What is the first decision to make to solve the problem?
  * For fibonacci: add fib(n-1) + fib(n-2)
* What subproblem(s) emerge out of that first decision?
  *   fib(n-1) and fib(n-2)
* You must wait for subproblem solutions to make the first decision

* Start with recursive solution
* If inefficient, do you have overlapping subproblms
* Identify unique subproblems
* Allocate array to hold their solutions
* Solve them from bottom-up smaller to larger
* Optimize space if possible

### Knapsack Problem
* A knapsack can hold a weight limit *L*
* A set of n item types
  * Each item has a weight (w<sub>i</sub>) and value (v<sub>i</sub>) 
  * Unbounded supply of all types
* Goal: Find the **maximum value** we can fit in the knapsack 
* ![image](https://user-images.githubusercontent.com/122314614/233852260-ddba6964-fc9b-4dd0-bfc4-3c53909d6bf2.png)

### Greedy Algorithms
* Greedy Algorithms are algorithms that make a choice that **seems to be the best at the moment. It does not wait for subproblem solutions*
* Examples
  * Huffamn Tries
  * Prim's MST 

#### Knapsack example for greedy algorithms
* Goal is to add as many copies of the **highest value per pound item** as possible
  * **Water: 30/6 = 5**
  * Rope: 14/3 = 4.66
  * Flashlight: 16/4 = 4
  * Moon Pie: 9/2 = 4.5 
* Highest value per pound item is water. Add that first
  * Can fit 1 item with 4 spaces left over
* Next highest value item: Rope
  * an fit 1 with 0 space left over
* No more room for anything else
* Total value in 10 lb. knapsack: 44    
* Greedy algorithm does not work for this problem
  * No optimal solution includes the locally optimal choices made by the greedy algorithm 

#### Knapsack recursive solution psuedocode
```java
int knapSack(int[] wt, int[] val, int L) {
  if (L == 0) { return 0 };
  int maxValue = 0;
  for(int i=0; i < n; i++){
    if (wt[i] <= L) {
      value = val[i] +
        knapSack(wt, val, L-wt[i]);
      if (value > maxValue) maxValue = value;
    }
  }
  return maxValue;
}
```
* Updated problem with recusrive solution
![image](https://user-images.githubusercontent.com/122314614/233852601-46f3f554-c4fe-4966-ac0c-7c37ac203991.png)
* Allocate a 2D array to hold all possible solutions
  * K[] with size L, the knapsack capacity
* Solve them from bottom-up smallest to largest
  * K[i] holds the mxaimum value possible with a knapsack of capacity *i*   

* Bottom-up solution
```
K[0] = 0
for (l = 1; l <= L; l++) {
  int max = 0;
  for (i = 0; i < n; i++) {
    if (wi <= l && vi + K[l - wi]) > max) {
      max = vi + K[l - wi];
    }
  }
  K[l] = max;
}
```
![image](https://user-images.githubusercontent.com/122314614/233852792-2d49d839-e178-4c28-ad46-18ddc7b7acdb.png)

### 0/1 Knapsack Problem
* A finite set of items each witha weight and value
  * Two choices for each item
    * goes in the knapsack or gets left out
* First decision
  * to place or not place the first item (or last item)
* item placed -> one less item and capacity less by item's wieight
* item not placed -> one less item and same capacity
* **Basically: we can't resue items**

#### 0/1 Psesudocode (Recursive)
```java
int knapSack(int[] wt, int[] val, int L, int n) {
  if (n == 0 || L == 0) { return 0 };
  //try placing the (n-1)st item
  if (wt[n-1] > L) { //cannot place
    return knapSack(wt, val, L, n-1)
} else {
  return max( val[n-1] + knapSack(wt, val, L-wt[n-1], n-1),
    knapSack(wt, val, L, n-1)
    );
  }
}
```
![image](https://user-images.githubusercontent.com/122314614/233853388-4d888ac2-c6dc-438f-bbc6-823005f11010.png)
#### 0/1 Pseudocode (Dynamic)
```java
int knapSack(int wt[], int val[], int L, int n) {
  int[][] K = new int[n+1][L+1];
  for (int i = 0; i <= n; i++) {
    for (int l = 0; l <= L; l++) {
      if (i==0 || l==0){ K[i][l] = 0 };
      //try to add item i-1
      else if (wt[i-1] > l){ K[i][l] = K[i-1][l] };
      else {
        K[i][l] = max(val[i-1] + K[i-1][l-wt[i-1]],
                  K[i-1][l]);
       }
    }
  }
  return K[n][L];
}
```

![image](https://user-images.githubusercontent.com/122314614/233853568-eec11eac-0759-4be9-b725-c3589271f157.png)
* **For this graph - Start at the first item (weight/value), find a spot on the column where the weight matche the column number and add the value to the row/col. Add up weights only using the current item and any ones before it**

### Change-Making Problem (Greedy)
* What is the minimum number of coins needed to make up a give change value **k >= 0**?
* Fore greedy algorithms to produce optimal results, problems must have two properties:
  * Optimal Substrucure: optimal solution to a subproblem leads to an optimal solution for the overall problem 
    * Best way to make change for 3 cents -> Best way to make change for 6 cents
  * Greedy Choice Property: Globally optimal solutions assembled from locally optimal choices
    * K = 6: for US corrency, the bset overall choice would be to use the biggest coin (nickel)
    * With thrickles/fourters, we can't know until we've looked at all possible breakdowns

### Subset Sum
* Problem: Given a set of **non-negative integers** S and a **target sum** k, is there a subset of S that sums to exactly k?
![image](https://user-images.githubusercontent.com/122314614/233854155-dfcc794c-fe6e-4db8-8d86-cb335adb96fe.png)

#### Subset sum recursive pseudocode (I realize that none of this is actually pseudocode but i'm to lazy to change these names)
```java
boolean SSS(int set[], int sum, int n) {
  //base cases
  if (sum == 0)
    return true;
  if (sum != 0 && n == 0)
    return false;
  //can we include item n-1?
  if (set[n-1] > sum)
    return SSS(set, sum, n-1);
  //should we include item n-1?
  return SSS(set, sum, n-1) ||
    SSS(set, sum-set[n-1], n-1);
}
```
![image](https://user-images.githubusercontent.com/122314614/233854265-bd41bc11-f855-48e8-9438-b1ba77ffe6a0.png)
* Subset[i][j] is true iff a subset of the first *i* items sums up to *j*
* **The next column in row i will be true until j > Sum of the set (i.e [1, 2, 3, 4], row is true until j = 11)

#### Subset sum bottom-up algorithm
```java
boolean SSS(int set[], int sum, int n) {
  boolean[][] subset = new boolean[n+1][sum+1];
  //easy cases
  for (int i = 0; i <= n; i++) subset[i][0] = true;
  for (int i = 1; i <= sum; i++) subset[0][i] = false;
  
  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= sum; j++) {
    if (j >= set[i-1])
      subset[i][j] = subset[i-1][j] ||
                     subset[i–1][j-set[i-1]];
    else subset[i][j] = subset[i-1][j];
    }
  }
  return subset[n][sum];
}
```

### Edit Distance
* Given two strings
  * a string S of length n
  * a string T of length m
* Find a minimum number of edits to convert S to T
  * Called Levenshtein Distance (LD)
* Possible Edits
  * Change a character
  * Delete a character
  * Insert a character

#### ED Example
* Example: "WEASEL" -> "SEASHELL"
  * For "WEASEL":
    * Change W to S
    * Add an H in position 5
    * Add an L in position 8
    * **Results in "SEASHELL"**
      * If we reverse the arguments, we get the same distance from T to S (but the edits may be different)

#### ED Pseudocode
* We want to calculate D(S, T) where n is the length of S and m is the length of T
  * If n = 0 //BASE CASE 1
    * Return m (m appends will create T from S)
  * else if (m = 0) //BASE CASE 2
    * return n (n deletes will create T from S)
  * else
    * Consider character n of S and character m of T 
* If characters match
  * Result is the same as for strings with last characters removed (since they match
  * **Return D(n-1, m-1)
  * Recursively solve the prolem with both strings one character smaller
* If characters do not match
  * Change X to Y, then recursively solve the same problem but with both strings one character smaller
  * **Return D(n-1, m-1) + 1**

* We do not know which of these gives the minimum distance until we try them all
* We must try all subproblems and choose the one that vies the minimum result  
      
![image](https://user-images.githubusercontent.com/122314614/233855099-86aaefb9-109f-4837-872f-e381e12d5d8e.png)
* D[i][j] is the minimum distance between the first i letters of S and the first j letters of i

* D[i, j] = D[i-1, j-1] if we have a match
* When we have a mismatch, minimum of the cells
  * D[i-1, j-1] + 1
    * Change char at this point in S
  * D[i-1, j] + 1
    * Delete a char from S
  * D[i, j-1] + 1
    * Append a char to S 
* **Edit Distance Runtime: Θ(mn)**

### Longest Common Subsequence
* Given two sequences, return the longest common subsequence
* Ex:
  * A **Q** S R **J** K **V** B **I**
  * **Q** B W F **J** **V** **I** T U  

#### LCS Code
```
int LCSLength(String x, String y) {
  int[][] m = new int[x.length + 1][y.length + 1];
  for (int i=0; i <= x.length; i++) {
    for (int j=0; j <= y.length; j++) {
      if (i == 0 || j == 0) m[i][j] = 0;
      if (x.charAt(i) == y.charAt(j))
        m[i][j] = m[i-1][j-1] + 1;
      else
        m[i][j] = max(m[i][j-1], m[i-1][j]);
    }
  }
  return m[x.length][y.length];
}
```

### Reinforcement Learning
* A type of machine learning
  * An agent (like a robot) learns an optimal policy only by getting rewards from the environment 

![image](https://user-images.githubusercontent.com/122314614/233856712-90c787c0-7048-4a1b-8df7-ffff2a906907.png)

* Markov Decision Process:
  * A set of states 
    * Maze locations, agent healt
  * A set of agent actions
    * Move left, move right
  * Probabilites of ending up in a state given a current state and an action
  * Reward function
  * Starting state    

* An agent policy determines the probability of taking an action given a state
* An optimal policy gives the maximum total reward
* Expected reward from an action depends on
  * immediate reward
  * values of states reachable through the action
  * Sum_{all states}:
    * Prob of reaching state * state value  

#### Using Dynamic Programming to Solve an MDP
* Data Structure: Array of state values
* Step 0: Start with an initial policy and initial state values
  * e.g., all actions equally likely and state values = 0
* Step 1: Compute expected state values
 * optional: iterate until values converge
* Step 2: Modify policy to take the best action with probability 1.0 (given the current state values)
* Repeat Step 1 and 2 until policy converges
## Lloyd's Algorithm For k-means and K-means++




 





  



