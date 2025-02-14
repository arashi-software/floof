# floof - SIMD-accelerated multithreaded fuzzy search thats fast as f*ck
## Getting started
Add the floof library to your project with nimble
```sh
nimble add floof
```
## Usage
Floof uses the SSE2 simd intrinsics, which come standard on all CPU's produced after 2003. Just in case though floof has a failsafe, non simd search proc. Usage is simple as floof does all the heavy lifting
```nim
import floof
import std/[sequtils, strutils]
let
  dictionary = toSeq(walkDir("/usr/share/applications/")).mapIt(
    it.path.splitPath().tail.replace(".desktop", "")
  )
  searchTerm = paramStr(1)
  
echo "Searching for: ", searchTerm
let results = search(searchTerm, dictionary) # Use foo's search function
for res in results:
  echo res.text, " (score: ", res.score.formatFloat(ffDecimal, 3), ")"
```
Make sure to compile with the `--threads:on` flag
