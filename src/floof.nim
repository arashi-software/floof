import std/[algorithm, sequtils, strutils, bitops, os, cpuinfo]
import nimsimd/[sse2]

var simd = false

when (defined(gcc) or defined(clang)) and not defined(noSIMD):
  {.passC: "-msse2".}
  simd = true
  
type
  SearchResult* = object
    text*: string
    score*: float32
  ThreadData = object
    query: string
    haystack: seq[string]
    startIdx, endIdx: int
    results: ptr seq[SearchResult]

  SearchError* = object of CatchableError

const VectorSize = 16 # SSE2 register size in bytes

func scoreMatchFallback*(query, text: string): float32 {.inline.} =
  ## Fallback search function without SIMD
  if query.len == 0 or text.len == 0:
    return 0.0

  let
    queryLower = query.toLowerAscii()
    textLower = text.toLowerAscii()

  var
    score = 0.0'f32
    lastMatchPos = -1
    searchPos = 0
    consecutiveMatches = 0

  # Match each query character
  for i, qChar in queryLower:
    var found = false
    for j in searchPos ..< textLower.len:
      if textLower[j] == qChar:
        # Base score for match
        score += 1.0

        # Boost score for matches at start of words
        if j == 0 or textLower[j - 1] == ' ':
          score += 2.0

        # Boost score for consecutive matches
        if lastMatchPos != -1 and j == lastMatchPos + 1:
          consecutiveMatches += 1
          score += float32(consecutiveMatches) * 0.5
        else:
          consecutiveMatches = 0

        # Boost score for case match
        if text[j] == query[i]:
          score += 0.5

        lastMatchPos = j
        searchPos = j + 1
        found = true
        break

    if not found:
      return 0.0

  result = score / float32(max(text.len, query.len))
  if lastMatchPos < text.len div 2:
    result *= 1.2

# SSE2-accelerated search
func findCharacterSSE2(pattern: char, text: string, startPos: int): int {.inline.} =
  if startPos >= text.len:
    return -1

  let patternVec = mm_set1_epi8(pattern.int8)
  var pos = startPos

  # Handle unaligned start
  while pos < text.len and (cast[uint](addr text[pos]) and 15) != 0:
    if text[pos] == pattern:
      return pos
    inc pos

  # Process 16 bytes at a time with SSE2
  while pos <= text.len - VectorSize:
    try:
      let textVec = mm_loadu_si128(cast[ptr M128i](unsafeAddr text[pos]))
      let mask = mm_cmpeq_epi8(textVec, patternVec)
      let bitMask = mm_movemask_epi8(mask)

      if bitMask != 0:
        return pos + countTrailingZeroBits(uint16(bitMask))
    except:
      # Fallback to regular search if SSE2 fails
      return text.find(pattern, pos)

    pos += VectorSize

  # Check remaining characters
  while pos < text.len:
    if text[pos] == pattern:
      return pos
    inc pos

  result = -1

func scoreMatchSSE2*(query, text: string): float32 {.inline.} =
  ## SSE2 accelerated scoring
  try:
    if query.len == 0 or text.len == 0:
      return 0.0

    let
      queryLower = query.toLowerAscii()
      textLower = text.toLowerAscii()

    var
      score = 0.0'f32
      lastMatchPos = -1
      searchPos = 0
      consecutiveMatches = 0

    # Match each query character
    for i, qChar in queryLower:
      let matchPos = findCharacterSSE2(qChar, textLower, searchPos)
      if matchPos == -1:
        return 0.0

      score += 1.0

      if matchPos == 0 or textLower[matchPos - 1] == ' ':
        score += 2.0

      if lastMatchPos != -1 and matchPos == lastMatchPos + 1:
        consecutiveMatches += 1
        score += float32(consecutiveMatches) * 0.5
      else:
        consecutiveMatches = 0

      if text[matchPos] == query[i]:
        score += 0.5

      lastMatchPos = matchPos
      searchPos = matchPos + 1

    result = score / float32(max(text.len, query.len))
    if lastMatchPos < text.len div 2:
      result *= 1.2
  except:
    # Fallback to non-SIMD version if anything fails
    result = scoreMatchFallback(query, text)

proc searchThread(data: ThreadData) {.thread.} =
  for i in data.startIdx ..< data.endIdx:
    if simd:
      let score = scoreMatchSSE2(data.query, data.haystack[i])
      if score > 0:
        data.results[].add(SearchResult(text: data.haystack[i], score: score))
    else:
      # Try fallback if SSE2 version fails
      let score = scoreMatchFallback(data.query, data.haystack[i])
      if score > 0:
        data.results[].add(SearchResult(text: data.haystack[i], score: score))

proc search*(query: string, haystack: seq[string]): seq[SearchResult] =
  ## SSE2 accelerated and multithreaded search
  runnableExamples:
    import floof
    import std/[sequtils, strutils]
    let
      dictionary = toSeq(walkDir("/usr/share/applications/")).mapIt(
        it.path.splitPath().tail.replace(".desktop", "")
      )
      searchTerm = paramStr(1)
  
    echo "Searching for: ", searchTerm
    let results = search(searchTerm, dictionary)
    for res in results:
      echo res.text, " (score: ", res.score.formatFloat(ffDecimal, 3), ")"
  if haystack.len == 0 or query.len == 0:
    return @[]

  let
    numThreads = countProcessors()
    chunkSize = max(256, haystack.len div numThreads)

  var
    threads: seq[Thread[ThreadData]]
    threadResults: seq[ptr seq[SearchResult]]

  threads.setLen(numThreads)
  threadResults.setLen(numThreads)

  # Create results sequences for each thread
  for i in 0 ..< numThreads:
    threadResults[i] = create(seq[SearchResult])
    threadResults[i][] = @[]

  # Spawn threads
  var startIdx = 0
  for i in 0 ..< numThreads:
    let endIdx = min(startIdx + chunkSize, haystack.len)
    if startIdx >= endIdx:
      break

    let threadData = ThreadData(
      query: query,
      haystack: haystack,
      startIdx: startIdx,
      endIdx: endIdx,
      results: threadResults[i],
    )

    createThread(threads[i], searchThread, threadData)
    startIdx += chunkSize

  # Wait for threads to complete
  for i in 0 ..< numThreads:
    if startIdx > i * chunkSize:
      joinThread(threads[i])

  # Combine results
  result = @[]
  for i in 0 ..< numThreads:
    if startIdx > i * chunkSize:
      result.add(threadResults[i][])
      dealloc(threadResults[i])

  # Sort results
  result.sort(
    proc(x, y: SearchResult): int =
      result = cmp(y.score, x.score)
      if result == 0:
        result = cmp(x.text.len, y.text.len)
      if result == 0:
        result = cmp(x.text, y.text)
  )
