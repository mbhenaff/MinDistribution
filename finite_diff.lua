require 'nn'

function gradPerturbationI(f, x, i, eps)
   local x1 = x:clone()
   local x2 = x:clone()
   x2[i] = x2[i]+eps
   x1[i] = x1[i]-eps
   local out = (f(x2)-f(x1))/(2.*eps)
   return out
end

function gradPerturbation(f, x, eps)
   local n = x:size(1)
   local out = torch.Tensor(n)
   for i = 1,n do
      out[i] = gradPerturbationI(f, x, i, eps)
   end
   return out
end

function hessianPerturbation(f, x, eps)
   local n = x:size(1)
   local out = torch.Tensor(n, n)
   for i = 1,n do
      local function g(x)
	 return gradPerturbationI(f, x, i, eps)
      end
      out[i]:copy(gradPerturbation(g, x, eps))
   end
   return out
end

function hessianPerturbationNet(f, x, eps)
   --eps = eps / x:norm()
   local n = x:size(1)
   local out = torch.Tensor(n, n)
   local x1 = x:clone()
   local x2 = x:clone()
   local gx1tmp = torch.Tensor(n)
   for i = 1,n do
      x1[i] = x1[i] + eps
      x2[i] = x2[i] - eps
      local _,gx1 = f(x1)
      gx1tmp:copy(gx1)
      local _,gx2 = f(x2)
      out[i]:copy((gx1tmp - gx2)/(2*eps))
      x1[i] = x1[i] - eps
      x2[i] = x2[i] + eps
   end
   return out
end

function hessianPerturbationNet2(f, x, eps)
   local n = x:size(1)
   local out = torch.Tensor(n, n)
   local x1 = x:clone()
   local _,gx = f(x)
   for i = 1,n do
      x1[i] = x1[i] + eps
      local _,gx1 = f(x1)
      out[i]:copy((gx1 - gx)/eps)
      x1[i] = x1[i] - eps
   end
   return out
end

function hessianPerturbationNetI(f, x, i, eps)
   local x1 = x:clone()
   local x2 = x:clone()
   x1[i] = x1[i] + eps
   x2[i] = x2[i] - eps
   local _,gx1 = f(x1)
   local _,gx2 = f(x2)
   return (gx1 - gx2)/(2*eps)
end

function gradPerturbation_testme()
   local net = nn.Sequential()
   net:add(nn.Linear(10,5))
   net:add(nn.Tanh())
   local params, gradParams = net:getParameters()
   gradParams:zero()
   local crit = nn.MSECriterion()
   local netoutput, neterr
   local x0 = torch.Tensor{0,2,1,4,3,-1,2,-4,-5,21}
   local target = torch.Tensor{1,2,3,4,5}
   local function f(w)
      local oldp = params:clone()
      params:copy(w)
      netoutput = net:forward(x0)
      neterr = crit:forward(netoutput, target)
      params:copy(oldp)
      return neterr
   end
   local gradPert = gradPerturbation(f, params, 1e-5)
   local df_do = crit:backward(netoutput, target)
   net:backward(x0, df_do)
   local diff = gradParams-gradPert
   if diff:abs():sum() > 1e-3 then
      print "gradPerturbation_testme : FAILED"
   else
      print "gradPerturbation_testme :  OK"
   end
end

function hessianPerturbation_testme()
   local net = nn.Sequential()
   net:add(nn.Linear(10,5))
   net:add(nn.Tanh())
   net:add(nn.Linear(5,1))
   local params, gradParams = net:getParameters()
   local x0 = torch.randn(10)
   local function f(w)
      local oldp = params:clone()
      params:copy(w)
      local out = net:forward(x0)
      params:copy(oldp)
      return out[1]
   end
   --print(f(params))
   local H = hessianPerturbation(f, params, 1e-5)
   local gx = gradPerturbation(f, params, 1e-5)
   local p = torch.randn(params:size(1))*0.001
   local gxp = gradPerturbation(f, params+p, 1e-5)
   local d = torch.Tensor(params:size(1), 2)
   d[{{},1}] = gx + torch.mm(p:resize(p:size(1), 1):t(), H) - gxp
   d[{{},2}] = gx - gxp
   if d[{{},1}]:abs():sum() > 0.01*d[{{},2}]:abs():sum() then
      print "hessianPerturbation_testme : FAILED"
   else
      print "hessianPerturbation_testme :  OK"
   end
end

function hessianPerturbation_checkHessianNet(f, H, params)
   local _,gx = f(params)
   local p = torch.randn(params:size(1))*0.001
   local _,gxp =f(params+p)
   local d = torch.Tensor(params:size(1), 2)
   d[{{},1}] = gx + torch.mm(p:resize(p:size(1), 1):t(), H) - gxp
   d[{{},2}] = gx - gxp
   if d[{{},1}]:abs():sum() > 0.01*d[{{},2}]:abs():sum() then
      print "hessianPerturbation_check : FAILED"
      exit(0)
   else
      print "hessianPerturbation_check :  OK"
   end
end

function hessian2_testme()
   gradPerturbation_testme()
   hessianPerturbation_testme()
end