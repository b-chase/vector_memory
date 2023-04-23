
import vector_memory as vm


x = vm.VMemory('blahblah', [1, 2, 3])
y = vm.VMemory('blehbleh', [1, 0, 3])

print(x)
print(y)

print(x.similarity(y))