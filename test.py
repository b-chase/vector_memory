
import vector_memory as vm


x = vm.Memory('blahblah', [1, 2, 3])
y = vm.Memory('blehbleh', [1, 0, 3])

print(x)
print(y)

s = vm.MemoryBank(3, [x, y])
# s.add_memories(x, y)

print(f"Memory store has {len(s)} memories")
for mem in s:
    print("Memories stored iterated: ",mem)


z = vm.Memory('teehee', [1, 0, 2.5])

print(s.search_memories_like(z, 2))