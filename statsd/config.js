{
  port: 8125,
  backends: ["./backends/graphite"],
  graphiteHost: "127.0.0.1",
  graphitePort: 2003,
  flushInterval: 10000,
  percentThreshold: [75, 90, 95, 99, 99.9]
}
