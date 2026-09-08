// A stand-in for a real RAG system: reads cases on stdin, writes predictions
// on stdout. Any language works as long as it speaks JSON lines.
let input = '';
process.stdin.on('data', (chunk) => (input += chunk)).on('end', () => {
  for (const line of input.split('\n').filter(Boolean)) {
    const testCase = JSON.parse(line);
    process.stdout.write(JSON.stringify({
      id: testCase.id,
      answer: testCase.expected_answer,
      retrieved: testCase.relevant_docs,
    }) + '\n');
  }
});
