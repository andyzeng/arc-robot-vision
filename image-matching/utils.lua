




-- Lookup filenames in directory (with search query string)
function scanDir(directory,query)
    local i, t, popen = 0, {}, io.popen
    local pfile = popen('ls -a "'..directory..'"')
    for filename in pfile:lines() do
        if filename == '.' or  filename == '..' then 
        else
            if query then
                if string.find(filename,query) then
                    i = i+1
                    t[i] = filename
                end
            else
                print(filename)
                i = i+1
                t[i] = filename
            end
        end
    end
    pfile:close()
    return t
end

-- Get subset of a 1D table
function subrange(t,first,last)
  local sub = {}
  for i = first,last do
    sub[#sub+1] = t[i]
  end
  return sub
end

-- Recursively freeze layers of the model
function recursiveModelFreeze(model)
    for i = 1,model:size() do
        local tmpLayer = model:get(i)
        if torch.type(tmpLayer):find('Convolution') or torch.type(tmpLayer):find('Linear') then

            -- Set parameter update functions to empty functions
            tmpLayer.accGradParameters = function() end
            tmpLayer.updateParameters = function() end
        end
        if torch.type(tmpLayer):find('Sequential') or torch.type(tmpLayer):find('ConcatTable') then
            recursiveModelFreeze(tmpLayer)
        end
    end
end

-- Load depth file (saved as 16-bit PNG in centimeters)
function loadDepth(filename)
    depth = image.load(filename)*65536/10000
    depth = depth:clamp(0.2,1.2) -- Depth range of Intel RealSense F200
    depth = depth:csub(0.440931) -- Subtract average mean depth value from training data
    return depth
end

-- Pre-process images for ResNet-101 pre-trained on ImageNet (224x224 RGB mean-subtracted)
function preprocessImg(img)
    img = image.scale(img,224,224)
    local mean = {0.485,0.456,0.406}
    local std = {0.229,0.224,0.225}
    for i=1,3 do
        img[i]:add(-mean[i])
        img[i]:div(std[i])
    end
    return img
end

-- Check if a file exists
function fileExists(file)
    local f = io.open(file, "rb")
    if f then f:close() end
    return f ~= nil
end

-- Get all lines from a file, returns an empty list/table if the file does not exist
function getLinesFromFile(file)
    if not fileExists(file) then return {} end
    lines = {}
    for line in io.lines(file) do 
        lines[#lines + 1] = line
    end
    return lines
end

-- Shuffle table
function shuffleTable(t,n)
    while n > 2 do -- only run if the table has more than 1 element
        local k = math.random(n) -- get a random number
        t[n], t[k] = t[k], t[n]
        n = n - 1
    end
    return t
end

-- -- Includes: matching tote image and product image, and non-matching tote image
-- function getTrainingExampleTriplet(productImgsPath,toteImgsPath,objNames)

--     -- Pick two different random objects
--     local randObjIdx1 = math.floor(math.random()*#objNames)+1
--     local randObjIdx2 = randObjIdx1
--     while randObjIdx2 == randObjIdx1 do
--         randObjIdx2 = math.floor(math.random()*#objNames)+1
--     end
--     local randObjName1 = objNames[randObjIdx1]
--     local randObjName2 = objNames[randObjIdx2]

--     -- Get product image of first random object
--     local prodImg = image.load(paths.concat(productImgsPath,randObjName1..'.jpg'))

--     -- Get a tote image of the first random object
--     local toteImgNames1 = scanDir(paths.concat(toteImgsPath,randObjName1),'.color.png')
--     local toteImgPos = image.load(paths.concat(toteImgsPath,randObjName1,toteImgNames1[math.floor(math.random()*#toteImgNames1)+1]))

--     -- Get a tote image of second random object
--     local toteImgNames2 = scanDir(paths.concat(toteImgsPath,randObjName2),'.color.png')
--     local toteImgNeg = image.load(paths.concat(toteImgsPath,randObjName2,toteImgNames2[math.floor(math.random()*#toteImgNames2)+1]))

--     -- Pre-process all images
--     prodImg = preprocessImg(prodImg)
--     toteImgPos = preprocessImg(toteImgPos)
--     toteImgNeg = preprocessImg(toteImgNeg)

--     return toteImgPos,prodImg,toteImgNeg
-- end

-- -- Includes: matching product image and tote image
-- function getTrainingExamplePositive(productImgsPath,toteImgsPath,objNames)

--     -- Get a random product image
--     local randObjIdx = math.floor(math.random()*#objNames)+1
--     local randObjName = objNames[randObjIdx]
--     local prodImg = image.load(paths.concat(productImgsPath,randObjName..'.jpg'))

--     -- Get a random tote image of the same object
--     local toteImgNames = scanDir(paths.concat(toteImgsPath,randObjName),'.color.png')
--     local toteImg = image.load(paths.concat(toteImgsPath,randObjName,toteImgNames[math.floor(math.random()*#toteImgNames)+1]))

--     -- Pre-process all images
--     prodImg = preprocessImg(prodImg)
--     toteImg = preprocessImg(toteImg)

--     return prodImg,toteImg
-- end

-- -- Includes: non-matching product image and tote image
-- function getTrainingExampleNegative(productImgsPath,toteImgsPath,objNames)

--     -- Pick two different random objects
--     local randObjIdx1 = math.floor(math.random()*#objNames)+1
--     local randObjIdx2 = randObjIdx1
--     while randObjIdx2 == randObjIdx1 do
--         randObjIdx2 = math.floor(math.random()*#objNames)+1
--     end
--     local randObjName1 = objNames[randObjIdx1]
--     local randObjName2 = objNames[randObjIdx2]

--     -- Get product image of first random object
--     local prodImg = image.load(paths.concat(productImgsPath,randObjName1..'.jpg'))

--     -- Get tote image of second random object
--     local toteImgNames = scanDir(paths.concat(toteImgsPath,randObjName2),'.color.png')
--     local toteImg = image.load(paths.concat(toteImgsPath,randObjName2,toteImgNames[math.floor(math.random()*#toteImgNames)+1]))

--     -- Pre-process all images
--     prodImg = preprocessImg(prodImg)
--     toteImg = preprocessImg(toteImg)

--     return prodImg,toteImg
-- end