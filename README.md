# 1501 Algorithms - Basically the canvas study guide but now I don't have to go through all the slides anymore

## Compression Algorithms

### LZW Compression
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

### LZW Expansion
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

### Expansion Corner Case
* Expansion can sometimes be a step ahead of compression
  * If during compression, the (pattern, codeword) that was **just added** to the directory is **immediately used** in the next step, the decompression algorithm will not yet know the codeword
  * This is easily detected and dealt with however (Use previous output + 1st character of the previous output)
  
### Adaptive LZW/Variable Width Codewords
* How long should codewords be?
  * Use fewer bites
    * Gives better compression earier on
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

### Move-To-Front Encoding
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

### Move-To-Front Encoding
* Same as encoding, except start from number and go to letters

### Burrows Wheeler Transform
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

### Burrows-Wheeler Transform Decoding
*How to recover original string? (ABABABA)
  1. Sort The encoded string (BBBAAAA -> AAABBB)
  2. Fill an array next[]
      * Defined for each entry in the sorted array
      * Holds the index in sorted array of the next string in the original array
      * Scan through the first column
        * For each row *i* holding character *c*
        * next[i] = first unassinged index *c* in the last column
  3. Recover the input string using next[] array
      
### Downsides of Burrows-Wheeler
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

### Adjacency Lists
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

### Graph Definitions
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
  
### Adjacency Matrix
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


  



