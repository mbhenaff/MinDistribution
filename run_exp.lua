require 'optim'
require 'nn'
require 'image'
require 'finite_diff'
require 'random'

printf = function(s,...)
	return io.write(s:format(...))
end

function eigIndex(eigenvals)
   local index = 0
   for i = 1,eigenvals:size(1) do
      if eigenvals[i] > 0 then
	 index = index + 1
      elseif eigenvals[i] < 0 then
	 index = index - 1
      end
   end
   return index
end


function analyze_model(model, criterion, nPtsPerDim, gridRange, data, gradThreshold)
   print ("analyze")
	local criticalPts={}
	local grid = torch.linspace(-gridRange/2,gridRange/2,nPtsPerDim)
	local param,gradParam=model:getParameters()
	local D=param:size(1)
	print("" ..D .. " dimensions")
	--local lossValues=torch.Tensor((torch.ones(D)*nPtsPerDim):long():storage())
	print(torch.prod(torch.ones(D)*nPtsPerDim, 1))
	--local gradValues=torch.Tensor((torch.ones(D)*nPtsPerDim):long():storage())
	local indx = torch.LongTensor(D)

	for i=0,nPtsPerDim^D-1 do
	   if i % 1000 == 0 then
	      xlua.progress(i,nPtsPerDim^D-1)
	   end
        	for j=1,D do
			indx[j] = math.mod(math.floor(i/nPtsPerDim^(j-1)),nPtsPerDim)+1
                	param[j] = grid[ indx[j] ]
		end
		gradParam:zero()
		local function feval(x)
			if x ~= param then
				param:copy(x)
			     end
			     --[[
			local loss=0
			for k=1,data[1]:size(1) do
				local input=data[1][k]
				local target=data[2][k]
				local output=model:forward(input)
				loss=loss+criterion:forward(output,target)
				local dL_do=criterion:backward(output,target)
				model:backward(input,dL_do)
			     end
			loss = loss/data[1]:size(1)
			gradParam:div(data[1]:size(1))
			--]]
			local input=data[1]
			local target=data[2]
			local output=model:forward(input)
			local loss=criterion:forward(output,target)
			local dL_do=criterion:backward(output,target)
			model:backward(input,dL_do)

			return loss, gradParam
		end
		loss,_ = feval(param)
		--lossValues[indx:storage()]=loss
		local gradNorm=gradParam:norm()
		--gradValues[indx:storage()]=gradNorm
		if gradNorm < gradThreshold then
		   hessian=hessianPerturbationNet(feval, param, 1e-3)
		   e=torch.eig(hessian)[{{},1}]
		   criticalPts[#criticalPts+1]={eigenvals=e,param=param:clone(),indx=indx:clone()}
		end
	end
	return lossValues,gradValues,criticalPts
end

nSamples=100

-- model
dimInputs = 2
nHidden = 2
model = nn.Sequential()
model:add(nn.Linear(dimInputs, nHidden))
model:add(nn.Threshold())
model:add(nn.Linear(nHidden, 1))
--local layer1 = nn.CMul(dimInputs)
--model:add(layer1)
--local layer2 = nn.Mul(dimInputs)
--layer2.weight = layer1.weight[{{1,2}}]
--layer2.gradWeight = layer1.gradWeight[{{1,2}}]
--model:add(layer2)

criterion=nn.MSECriterion()

-- generate points
--data=torch.randn(1,nSamples,dimInputs):repeatTensor(2,1,1)
local nClass1 = math.floor(nSamples/2)
local nClass2 = math.ceil(nSamples/2)
data_class1 = sampleBallN(dimInputs, nClass1)
data_class2 = sampleSphereN(dimInputs, nClass2) * 5
data = {torch.Tensor(nSamples, dimInputs), torch.Tensor(nSamples, 1)}
data[1][{{1,nClass1},{}}]:copy(data_class1)
data[1][{{nClass1+1,nSamples},{}}]:copy(data_class2)
data[2][{{1,nClass1},{}}]:fill(-1)
data[2][{{nClass1+1,nSamples},{}}]:fill(1)

loss,gradNorm,criticalPts=analyze_model(model,criterion,2,4,data,0.3)

-- save data
local filenamebase = 'criticalPts_' .. nHidden .. 'hidden_'
local filenamenumber = 1
while paths.filep(filenamebase..filenamenumber..'.th') do
   filenamenumber = filenamenumber + 1
end
torch.save(filenamebase..filenamenumber..'.th',
	   {criticalPts=criticalPts,grid=grid,model=model,dataset=data, criterion=criterion})

--[[
-- display
gradNorm:reshape(gradNorm:nElement())
gnuplot.hist(gradNorm,100)

imdisp = torch.Tensor(3,loss:size(1), loss:size(2))
for i = 1,3 do
   imdisp[i]:copy(loss)
end


local maxLossDisplay = 3
for i = 1,#criticalPts do
   local indx = criticalPts[i]['indx']
   local eigenvals = criticalPts[i]['eigenvals']
   eigidx = eigIndex(eigenvals)
   eigidx_normalized = (eigidx + eigenvals:size(1))/(eigenvals:size(1)*2)
   imdisp[1][indx:storage()] = maxLossDisplay*eigidx_normalized
   imdisp[2][indx:storage()] = maxLossDisplay*(1-eigidx_normalized)
   imdisp[3][indx:storage()] = 0
end
image.display{image=imdisp, zoom=5, max=maxLossDisplay}
--]]