require 'math'
require 'xlua'

function sampleBall(dim)
   local p = torch.rand(dim) * 2 - 1
   while p:norm() > 1 do
      p = torch.rand(dim) * 2 - 1
   end
   return p
end

function sampleBallN(dim, nPoints)
   local output = torch.Tensor(nPoints, dim)
   for i = 1,nPoints do
      output[i] = sampleBall(dim)
   end
   return output
end

function sampleSphere(dim)
   local p = sampleBall(dim)
   while p:norm() < 1e-4 do
      p = sampleBall(dim)
   end
   return p / p:norm()
end

function sampleSphereN(dim, nPoints)
   local output = torch.Tensor(nPoints, dim)
   for i = 1,nPoints do
      output[i] = sampleSphere(dim)
   end
   return output
end