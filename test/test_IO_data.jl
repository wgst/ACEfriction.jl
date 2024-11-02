using ACEfriction
using ACEfriction.DataUtils: load_h5fdata, save_h5fdata
using Test
using ACEbase.Testing

@info "Test HDF5 import and export of friction data."  

fname = "/test/test-data-100"
filename = string(pkgdir(ACEfriction),fname,".h5")

data1 = ACEfriction.DataUtils.load_h5fdata(filename); 
save_h5fdata(data1, "./test/test-data-temp.h5");
data2= load_h5fdata("./test/test-data-temp.h5");
rm("./test/test-data-temp.h5")

@test all([all([getfield(d1.at,f) == getfield(d2.at,f) && 
typeof(getfield(d1.at,f)) == typeof(getfield(d2.at,f)) for f in fieldnames(typeof(d1.at))])
for (d1,d2) in zip(data1,data2)])