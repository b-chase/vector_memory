
import vector_memory as vm


x = vm.Memory('blahblah', [1, 2, 3])
y = vm.Memory('blehbleh', [1, 0, 3])

print(x)
print(y)

s = vm.MemoryBank(3)
s.add_memories(x, y)

for mem in s:
    print("Memories stored iterated: ",mem)
print(f"Memory store has {len(s)} memories")


z = vm.Memory('teehee', [1, 0, 2.5])

print(s.search_memories_like(z, 2))