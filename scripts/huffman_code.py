"""
-----------------------------------------------------------------------
🗜️ Huffman Coding Implementation in Python - Compression Module
-----------------------------------------------------------------------

This script implements Huffman Coding — a popular lossless data compression
algorithm. It includes the full process of:

1. Calculating character frequencies
2. Building a Huffman tree using a custom min-heap
3. Generating binary codes for each character
4. Compressing a string into a binary representation
5. Decompressing the binary string back to its original form

✅ This module is used to compute compression ratio for email text in
spam detection tasks, aiding in feature engineering.

📦 Functions:
- calculate_frequencies
- build_huffman_tree
- generate_codes
- compress
- decompress
- huffman_coding

By Adel Muhammad Haiba | CS Student | Data Science & ML Enthusiast
"""

def calculate_frequencies(text):
    freq = {}
    for char in text:
        if char not in freq:
            freq[char] = 0
        freq[char] += 1
    return freq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

class MinHeap:
    def __init__(self):
        self.heap = []
    
    def insert(self, node):
        self.heap.append(node)
        self._sift_up(len(self.heap) - 1)
    
    def extract_min(self):
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        
        min_node = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return min_node
    
    def _sift_up(self, index):
        parent = (index - 1) // 2
        if index > 0 and self.heap[index] < self.heap[parent]:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            self._sift_up(parent)
    
    def _sift_down(self, index):
        min_index = index
        left = 2 * index + 1
        right = 2 * index + 2
        
        if left < len(self.heap) and self.heap[left] < self.heap[min_index]:
            min_index = left
        if right < len(self.heap) and self.heap[right] < self.heap[min_index]:
            min_index = right
        
        if min_index != index:
            self.heap[index], self.heap[min_index] = self.heap[min_index], self.heap[index]
            self._sift_down(min_index)

def build_huffman_tree(freq):
    heap = MinHeap()
    
    for char, f in freq.items():
        heap.insert(Node(char, f))
    
    while len(heap.heap) > 1:
        left = heap.extract_min()
        right = heap.extract_min()
        
        internal = Node(None, left.freq + right.freq)
        internal.left = left
        internal.right = right
        heap.insert(internal)
    
    return heap.extract_min()

def generate_codes(root, current_code="", codes=None):
    if codes is None:
        codes = {}
    
    if root is None:
        return
    
    if root.char is not None:
        codes[root.char] = current_code or "0"
    
    generate_codes(root.left, current_code + "0", codes)
    generate_codes(root.right, current_code + "1", codes)
    
    return codes

def compress(text, codes):
    encoded_text = ""
    for char in text:
        encoded_text += codes[char]
    return encoded_text

def decompress(encoded_text, root):
    decoded_text = ""
    current = root
    
    for bit in encoded_text:
        if bit == "0":
            current = current.left
        else:
            current = current.right
        
        if current.char is not None:
            decoded_text += current.char
            current = root
    
    return decoded_text

def huffman_coding(text):
    if not text:
        return "", None
    
    freq = calculate_frequencies(text)
    
    root = build_huffman_tree(freq)
    
    codes = generate_codes(root)
    
    encoded_text = compress(text, codes)
    
    decoded_text = decompress(encoded_text, root)
    
    return encoded_text, decoded_text, codes

if __name__ == "__main__":
    text = "huffman coding"
    encoded, decoded, codes = huffman_coding(text)
    
    print(f"Original text: {text}")
    print(f"Encoded text: {encoded}")
    print(f"Decoded text: {decoded}")
    print(f"Huffman Codes: {codes}")
