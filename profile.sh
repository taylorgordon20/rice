env CPUPROFILE=rt.prof ./rt 
env PATH=$PATH:$(brew --prefix llvm)/bin/llvm-symbolizer pprof --text rt rt.prof
