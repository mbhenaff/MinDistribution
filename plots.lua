require 'gnuplot'
require 'optim'
require 'nn'
require 'image'

data = torch.load('criticalPts_2hidden.th')
criticalPts = data['criticalPts']
grid = data['grid']
model = data['model']
criterion = data['criterion']
dataset = data['dataset']
eps = 1e-2

nIdx = 2*criticalPts[1]['eigenvals']:size(1)+1
indices = torch.Tensor(#criticalPts)
loss = torch.Tensor(#criticalPts)

local param,gradParam=model:getParameters()
local D = param:size(1)
local input = dataset[1]
local target = dataset[2]

for i = 1,#criticalPts do
   local e = criticalPts[i]['eigenvals']
   local indx = criticalPts[i]['indx']
   local idx = 0
   for j = 1,e:size(1) do
      if e[j] > eps then
	 idx = idx + 1
      else
	 if e[j] < -eps then
	    idx = idx - 1
	 end
      end
   end
   indices[i] = idx
   for j=1,D do
      param[j] = grid[ indx[j] ]
   end
   loss[i] = criterion:forward(model:forward(input), target)
end
gnuplot.figure(1)
gnuplot.hist(indices, nIdx)
gnuplot.figure(2)
gnuplot.hist(loss, 100)
local ifig = 3

for iIdx = -criticalPts[1]['eigenvals']:size(1),criticalPts[1]['eigenvals']:size(1) do
   lossIdx = loss[indices:eq(iIdx)]
   if lossIdx:nDimension() > 0 then
      gnuplot.figure(ifig)
      ifig = ifig + 1
      gnuplot.title('Index = ' .. iIdx)
      gnuplot.hist(lossIdx,100)
   end
end