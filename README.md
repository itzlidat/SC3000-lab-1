# SC3000 Lab Assignment 1

This repository contains my solution for **SC3000 / CZ3005 Lab Assignment 1**.

## Overview

This assignment has 2 main parts:

### Part 1: Graph Search
Using the provided NYC road network data:
- `G.json`
- `Dist.json`
- `Cost.json`
- `Coord.json`

the program solves:

1. **Task 1**: Shortest path from node `1` to node `50` without energy constraint  
2. **Task 2**: Uninformed search with energy budget constraint  
3. **Task 3**: A* search with energy budget constraint  

For each task, the program outputs:
- shortest path
- total distance
- total energy cost

## Files

- `main.py` — main program for Part 1
- `G.json` — graph adjacency list
- `Dist.json` — distance of each edge
- `Cost.json` — energy cost of each edge
- `Coord.json` — coordinates of each node
- `CZ3005 Lab Assignment 1(1).pdf` — assignment brief

## How to Run

Make sure Python 3 is installed, then run:

```bash
python main.py