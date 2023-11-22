TEXTURE2D(_Control0);

#define DECLARE_TERRAIN_LAYER_TEXS(n)   \
    TEXTURE2D(_Splat##n);               \
    TEXTURE2D(_Normal##n);              \
    TEXTURE2D(_Mask##n)                 

DECLARE_TERRAIN_LAYER_TEXS(0);
DECLARE_TERRAIN_LAYER_TEXS(1);
DECLARE_TERRAIN_LAYER_TEXS(2);
DECLARE_TERRAIN_LAYER_TEXS(3);
#ifdef _TERRAIN_8_LAYERS
    DECLARE_TERRAIN_LAYER_TEXS(4);
    DECLARE_TERRAIN_LAYER_TEXS(5);
    DECLARE_TERRAIN_LAYER_TEXS(6);
    DECLARE_TERRAIN_LAYER_TEXS(7);
    TEXTURE2D(_Control1);
#endif

#undef DECLARE_TERRAIN_LAYER_TEXS

SAMPLER(sampler_Splat0);
SAMPLER(sampler_Control0);

//sheep yang
#pragma multi_compile _ _NOTILE
#pragma multi_compile _ _TRIPLANAR
#pragma multi_compile _ _USEPIXELDISPLACEMENT
float _usePixelDisplacementHeight;
float _usePixelDisplacementSteps;
float _usePixelDisplacementRange;

float GetSumHeight(float4 heights0, float4 heights1)
{
    float sumHeight = heights0.x;
    sumHeight += heights0.y;
    sumHeight += heights0.z;
    sumHeight += heights0.w;
    #ifdef _TERRAIN_8_LAYERS
        sumHeight += heights1.x;
        sumHeight += heights1.y;
        sumHeight += heights1.z;
        sumHeight += heights1.w;
    #endif
    return sumHeight;
}

float3 SampleNormalGrad(TEXTURE2D_PARAM(textureName, samplerName), float2 uv, float2 dxuv, float2 dyuv, float scale)
{
    float4 nrm = SAMPLE_TEXTURE2D_GRAD(textureName, samplerName, uv, dxuv, dyuv);
#ifdef SURFACE_GRADIENT
    #ifdef UNITY_NO_DXT5nm
        return float3(UnpackDerivativeNormalRGB(nrm, scale), 0);
    #else
        return float3(UnpackDerivativeNormalRGorAG(nrm, scale), 0);
    #endif
#else
    #ifdef UNITY_NO_DXT5nm
        return UnpackNormalRGB(nrm, scale);
    #else
        return UnpackNormalmapRGorAG(nrm, scale);
    #endif
#endif
}

float4 RemapMasks(float4 masks, float blendMask, float4 remapOffset, float4 remapScale)
{
    float4 ret = masks;
    ret.b *= blendMask; // height needs to be weighted before remapping
    ret = ret * remapScale + remapOffset;
    return ret;
}

#ifdef OVERRIDE_SPLAT_SAMPLER_NAME
    #define sampler_Splat0 OVERRIDE_SPLAT_SAMPLER_NAME
    SAMPLER(OVERRIDE_SPLAT_SAMPLER_NAME);
#endif

void TerrainSplatBlend(float2 controlUV, float2 splatBaseUV, inout TerrainLitSurfaceData surfaceData)
{
    // TODO: triplanar
    // TODO: POM

    float4 albedo[_LAYER_COUNT];
    float3 normal[_LAYER_COUNT];
    float4 masks[_LAYER_COUNT];

#ifdef _NORMALMAP
    #define SampleNormal(i) SampleNormalGrad(_Normal##i, sampler_Splat0, splatuv, splatdxuv, splatdyuv, _NormalScale##i)
#else
    #define SampleNormal(i) float3(0, 0, 0)
#endif

#define DefaultMask(i) float4(_Metallic##i, _MaskMapRemapOffset##i.y + _MaskMapRemapScale##i.y, _MaskMapRemapOffset##i.z + 0.5 * _MaskMapRemapScale##i.z, albedo[i].a * _Smoothness##i)

#ifdef _MASKMAP
    #define MaskModeMasks(i, blendMask) RemapMasks(SAMPLE_TEXTURE2D_GRAD(_Mask##i, sampler_Splat0, splatuv, splatdxuv, splatdyuv), blendMask, _MaskMapRemapOffset##i, _MaskMapRemapScale##i)
#define SampleMasks(i, blendMask) lerp(DefaultMask(i), MaskModeMasks(i, blendMask), _LayerHasMask##i)
    #define NullMask(i)               float4(0, 1, _MaskMapRemapOffset##i.z, 0) // only height matters when weight is zero.
#else
    #define SampleMasks(i, blendMask) DefaultMask(i)
    #define NullMask(i)               float4(0, 1, 0, 0)
#endif

#define SampleResults(i, mask)                                                                                  \
    UNITY_BRANCH if (mask > 0)                                                                                  \
    {                                                                                                           \
        float2 splatuv = splatBaseUV * _Splat##i##_ST.xy + _Splat##i##_ST.zw;                                   \
        float2 splatdxuv = dxuv * _Splat##i##_ST.x;                                                             \
        float2 splatdyuv = dyuv * _Splat##i##_ST.y;                                                             \
        albedo[i] = SAMPLE_TEXTURE2D_GRAD(_Splat##i, sampler_Splat0, splatuv, splatdxuv, splatdyuv);            \
        albedo[i].rgb *= _DiffuseRemapScale##i.xyz;                                                             \
        normal[i] = SampleNormal(i);                                                                            \
        masks[i] = SampleMasks(i, mask);                                                                        \
    }                                                                                                           \
    else                                                                                                        \
    {                                                                                                           \
        albedo[i] = float4(0, 0, 0, 0);                                                                         \
        normal[i] = float3(0, 0, 0);                                                                            \
        masks[i] = NullMask(i);                                                                                 \
    }

    float2 dxuv = ddx(splatBaseUV);
    float2 dyuv = ddy(splatBaseUV);

    float2 blendUV0 = (controlUV.xy * (_Control0_TexelSize.zw - 1.0f) + 0.5f) * _Control0_TexelSize.xy;
    float4 blendMasks0 = SAMPLE_TEXTURE2D(_Control0, sampler_Control0, blendUV0);
    #ifdef _TERRAIN_8_LAYERS
        float2 blendUV1 = (controlUV.xy * (_Control1_TexelSize.zw - 1.0f) + 0.5f) * _Control1_TexelSize.xy;
        float4 blendMasks1 = SAMPLE_TEXTURE2D(_Control1, sampler_Control0, blendUV1);
    #else
        float4 blendMasks1 = float4(0, 0, 0, 0);
    #endif

    SampleResults(0, blendMasks0.x);
    SampleResults(1, blendMasks0.y);
    SampleResults(2, blendMasks0.z);
    SampleResults(3, blendMasks0.w);
    #ifdef _TERRAIN_8_LAYERS
        SampleResults(4, blendMasks1.x);
        SampleResults(5, blendMasks1.y);
        SampleResults(6, blendMasks1.z);
        SampleResults(7, blendMasks1.w);
    #endif

#undef SampleNormal
#undef SampleMasks
#undef SampleResults

    float weights[_LAYER_COUNT];
    ZERO_INITIALIZE_ARRAY(float, weights, _LAYER_COUNT);

    #ifdef _MASKMAP
        #if defined(_TERRAIN_BLEND_HEIGHT)
            // Modify blendMask to take into account the height of the layer. Higher height should be more visible.
            float maxHeight = masks[0].z;
            maxHeight = max(maxHeight, masks[1].z);
            maxHeight = max(maxHeight, masks[2].z);
            maxHeight = max(maxHeight, masks[3].z);
            #ifdef _TERRAIN_8_LAYERS
                maxHeight = max(maxHeight, masks[4].z);
                maxHeight = max(maxHeight, masks[5].z);
                maxHeight = max(maxHeight, masks[6].z);
                maxHeight = max(maxHeight, masks[7].z);
            #endif

            // Make sure that transition is not zero otherwise the next computation will be wrong.
            // The epsilon here also has to be bigger than the epsilon in the next computation.
            float transition = max(_HeightTransition, 1e-5);

            // The goal here is to have all but the highest layer at negative heights, then we add the transition so that if the next highest layer is near transition it will have a positive value.
            // Then we clamp this to zero and normalize everything so that highest layer has a value of 1.
            float4 weightedHeights0 = { masks[0].z, masks[1].z, masks[2].z, masks[3].z };
            weightedHeights0 = weightedHeights0 - maxHeight.xxxx;
            // We need to add an epsilon here for active layers (hence the blendMask again) so that at least a layer shows up if everything's too low.
            weightedHeights0 = (max(0, weightedHeights0 + transition) + 1e-6) * blendMasks0;

            #ifdef _TERRAIN_8_LAYERS
                float4 weightedHeights1 = { masks[4].z, masks[5].z, masks[6].z, masks[7].z };
                weightedHeights1 = weightedHeights1 - maxHeight.xxxx;
                weightedHeights1 = (max(0, weightedHeights1 + transition) + 1e-6) * blendMasks1;
            #else
                float4 weightedHeights1 = { 0, 0, 0, 0 };
            #endif

            // Normalize
            float sumHeight = GetSumHeight(weightedHeights0, weightedHeights1);
            blendMasks0 = weightedHeights0 / sumHeight.xxxx;
            #ifdef _TERRAIN_8_LAYERS
                blendMasks1 = weightedHeights1 / sumHeight.xxxx;
            #endif
        #elif defined(_TERRAIN_BLEND_DENSITY)
            // Denser layers are more visible.
            float4 opacityAsDensity0 = saturate((float4(albedo[0].a, albedo[1].a, albedo[2].a, albedo[3].a) - (float4(1.0, 1.0, 1.0, 1.0) - blendMasks0)) * 20.0); // 20.0 is the number of steps in inputAlphaMask (Density mask. We decided 20 empirically)
            opacityAsDensity0 += 0.001f * blendMasks0;      // if all weights are zero, default to what the blend mask says
            float4 useOpacityAsDensityParam0 = { _DiffuseRemapScale0.w, _DiffuseRemapScale1.w, _DiffuseRemapScale2.w, _DiffuseRemapScale3.w }; // 1 is off
            blendMasks0 = lerp(opacityAsDensity0, blendMasks0, useOpacityAsDensityParam0);
            #ifdef _TERRAIN_8_LAYERS
                float4 opacityAsDensity1 = saturate((float4(albedo[4].a, albedo[5].a, albedo[6].a, albedo[7].a) - (float4(1.0, 1.0, 1.0, 1.0) - blendMasks1)) * 20.0); // 20.0 is the number of steps in inputAlphaMask (Density mask. We decided 20 empirically)
                opacityAsDensity1 += 0.001f * blendMasks1;  // if all weights are zero, default to what the blend mask says
                float4 useOpacityAsDensityParam1 = { _DiffuseRemapScale4.w, _DiffuseRemapScale5.w, _DiffuseRemapScale6.w, _DiffuseRemapScale7.w };
                blendMasks1 = lerp(opacityAsDensity1, blendMasks1, useOpacityAsDensityParam1);
            #endif

            // Normalize
            float sumHeight = GetSumHeight(blendMasks0, blendMasks1);
            blendMasks0 = blendMasks0 / sumHeight.xxxx;
            #ifdef _TERRAIN_8_LAYERS
                blendMasks1 = blendMasks1 / sumHeight.xxxx;
            #endif
        #endif // if _TERRAIN_BLEND_HEIGHT
    #endif // if _MASKMAP

    weights[0] = blendMasks0.x;
    weights[1] = blendMasks0.y;
    weights[2] = blendMasks0.z;
    weights[3] = blendMasks0.w;
    #ifdef _TERRAIN_8_LAYERS
        weights[4] = blendMasks1.x;
        weights[5] = blendMasks1.y;
        weights[6] = blendMasks1.z;
        weights[7] = blendMasks1.w;
    #endif

    surfaceData.albedo = 0;
    surfaceData.normalData = 0;
    float3 outMasks = 0;
    UNITY_UNROLL for (int i = 0; i < _LAYER_COUNT; ++i)
    {
        surfaceData.albedo += albedo[i].rgb * weights[i];
        surfaceData.normalData += normal[i].rgb * weights[i]; // no need to normalize
        outMasks += masks[i].xyw * weights[i];
    }
    surfaceData.smoothness = outMasks.z;
    surfaceData.metallic = outMasks.x;
    surfaceData.ao = outMasks.y;
}

void TerrainLitShade(float2 uv, inout TerrainLitSurfaceData surfaceData)
{
    TerrainSplatBlend(uv, uv, surfaceData);
}

//sheep yang
float4 hash4(float2 src)
{
    float t1 = 1.0 +dot(src,float2(37,17));
    float t2 = 2.0 +dot(src,float2(11,47));
    float t3 = 3.0 +dot(src,float2(41,29));
    float t4 = 4.0 +dot(src,float2(23,31));
    return frac(sin(float4(t1,t2,t3,t4)) * 103);
}

float4 texNoTile(Texture2D tex, sampler samp, float2 uv)
{
    float2 iuv = floor(uv);//ensure which tile
    float2 fuv = frac(uv);//position of each tile

    float4 ofa = hash4(iuv + float2(0.0, 0.0));
    float4 ofb = hash4(iuv + float2(1.0, 0.0));
    float4 ofc = hash4(iuv + float2(0.0, 1.0));
    float4 ofd = hash4(iuv + float2(1.0, 1.0));

    float2 dx = ddx(uv);
    float2 dy = ddy(uv);

    //mirror
    // ofa.zw = sign(ofa.zw - 0.5);
    // ofb.zw = sign(ofb.zw - 0.5);
    // ofc.zw = sign(ofc.zw - 0.5);
    // ofd.zw = sign(ofd.zw - 0.5);

    // float2 uva = uv * ofa.zw + ofa.xy, dxa = dx * ofa.zw, dya = dy * ofa.zw;
    // float2 uvb = uv * ofb.zw + ofb.xy, dxb = dx * ofb.zw, dyb = dy * ofb.zw;
    // float2 uvc = uv * ofc.zw + ofc.xy, dxc = dx * ofc.zw, dyc = dy * ofc.zw;
    // float2 uvd = uv * ofd.zw + ofd.xy, dxd = dx * ofd.zw, dyd = dy * ofd.zw;

    float2 uva = uv + ofa.xy, dxa = dx , dya = dy ;
    float2 uvb = uv + ofb.xy, dxb = dx , dyb = dy ;
    float2 uvc = uv + ofc.xy, dxc = dx , dyc = dy ;
    float2 uvd = uv + ofd.xy, dxd = dx , dyd = dy ;

    float2 b = smoothstep(0.25, 0.75, fuv);

    float4 sampler1 = SAMPLE_TEXTURE2D_GRAD(tex,samp, uva, dxa, dya);
    float4 sampler2 = SAMPLE_TEXTURE2D_GRAD(tex,samp, uvb, dxb, dyb);
    float4 sampler3 = SAMPLE_TEXTURE2D_GRAD(tex,samp, uvc, dxc, dyc);
    float4 sampler4 = SAMPLE_TEXTURE2D_GRAD(tex,samp, uvd, dxd, dyd);

    return lerp(lerp(sampler1,sampler2,b.x),lerp(sampler3,sampler4,b.x),b.y);
}

float3 PomMapping(real lodThreshold, int numSteps, real3 viewDirTS, TEXTURE2D_PARAM(mask, samplerMask),float2 uv,float lod)
{
#ifdef _NOTILE
    numSteps = 8;   
#endif
    real stepSize = 1.0 / (real)numSteps;
    real2 parallaxMaxOffsetTS = viewDirTS.xy / -viewDirTS.z;
    real2 texOffsetPerStep = stepSize * parallaxMaxOffsetTS;

    real2 texOffsetCurrent = real2(0.0, 0.0);

    real prevHeight;
    real currHeight;
#ifdef _NOTILE
    prevHeight = texNoTile(mask, samplerMask, uv + texOffsetCurrent).b;
#else
    prevHeight = SAMPLE_TEXTURE2D_LOD(mask, samplerMask, uv + texOffsetCurrent,lod).b;
#endif
    texOffsetCurrent += texOffsetPerStep;
#ifdef _NOTILE
    currHeight = texNoTile(mask, samplerMask, uv + texOffsetCurrent).b;
#else
    currHeight = SAMPLE_TEXTURE2D_LOD(mask, samplerMask, uv + texOffsetCurrent,lod).b;
#endif
    real rayHeight = 1.0 - stepSize;


    for (int stepIndex = 0; stepIndex < numSteps; ++stepIndex)
    {
        if (currHeight > rayHeight)
          break; 
        prevHeight = currHeight;
        rayHeight -= stepSize;
        texOffsetCurrent += texOffsetPerStep;
#ifdef _NOTILE
        currHeight = texNoTile(mask, samplerMask, uv + texOffsetCurrent).b;
#else
        currHeight = SAMPLE_TEXTURE2D_LOD(mask, samplerMask, uv + texOffsetCurrent,lod).b;
#endif
    }

    real pt0 = rayHeight + stepSize;
    real pt1 = rayHeight;
    real delta0 = pt0 - prevHeight;
    real delta1 = pt1 - currHeight;

    real delta;
    real2 offset;


    for (int i = 0; i < 3; ++i)
    {
        real intersectionHeight = (pt0 * delta1 - pt1 * delta0) / (delta1 - delta0);
        offset = (1 - intersectionHeight) * texOffsetPerStep * numSteps;
#ifdef _NOTILE
        currHeight = texNoTile(mask, samplerMask, uv + offset).b;
#else
        currHeight = SAMPLE_TEXTURE2D_LOD(mask, samplerMask, uv + offset,lod).b;
#endif
        delta = intersectionHeight - currHeight;

        if (abs(delta) <= 0.01)
            break;

        if (delta < 0.0)
        {
            delta1 = delta;
            pt1 = intersectionHeight;
        }
        else
        {
            delta0 = delta;
            pt0 = intersectionHeight;
        }
    }

    return float3(currHeight,offset);
}


float ApplyPerPixelDisplacement(float3x3 tbn, float3 V, float2 uv,float2 xz,float2 xy,float2 zy,float wScale,float4 texelSize,float4 texelST, TEXTURE2D_PARAM(mask, samplerMask),float3 node_Blend,inout UVS uvs)
{
#ifdef _USEPIXELDISPLACEMENT
    float  maxHeight = abs(_usePixelDisplacementHeight * 0.001f);
    float2 minUvSize = float2(FLT_MAX, FLT_MAX);
    float worldScale = 1;

#ifdef _TRIPLANAR
    {
        minUvSize = min(zy * texelSize.zw, minUvSize);
        minUvSize = min(xz * texelSize.zw, minUvSize);
        minUvSize = min(xy * texelSize.zw, minUvSize);
        worldScale = wScale;
    }
#else
    {
        minUvSize = min(uv * texelSize.zw, minUvSize);
        worldScale = 1;
    }
#endif

    float2 DDX = ddx(minUvSize);
    float2 DDY = ddy(minUvSize);
    float d = max(dot(DDX, DDX), dot(DDY, DDY));
    float lod = max(0.5 * log2(d), 0.0);

    float2 uvSpaceScale = texelST.xy * maxHeight;
    float height = 0; 
    float NdotV  = 0;
    float3 final = float3(0,0,0);

#ifdef _TRIPLANAR
    {
        UNITY_BRANCH if (node_Blend.x >= 0.001)
        {
            float3 viewDirTS = float3(V.z,V.y,abs(V.x));
            float3 viewDirUV = normalize(float3(viewDirTS.xy * uvSpaceScale, viewDirTS.z)); 
            float  unitAngle = saturate(FastACosPos(viewDirUV.z) * INV_HALF_PI); 
            int    numSteps  = (int)lerp(_usePixelDisplacementSteps - 10, _usePixelDisplacementSteps, unitAngle);
            final    = PomMapping(_usePixelDisplacementRange,numSteps,viewDirUV,mask, samplerMask,zy,lod);

            uvs.uvzyOffset = final.yz;
            height += node_Blend.x * final.x;
            NdotV  += node_Blend.x * viewDirTS.z;
        }

        UNITY_BRANCH if (node_Blend.y >= 0.001)
        {
            float3 viewDirTS = float3(V.x,V.z,abs(V.y));
            float3 viewDirUV = normalize(float3(viewDirTS.xy * uvSpaceScale, viewDirTS.z));
            float  unitAngle = saturate(FastACosPos(viewDirUV.z) * INV_HALF_PI); 
            int    numSteps  = (int)lerp(_usePixelDisplacementSteps - 10, _usePixelDisplacementSteps, unitAngle);
            final    = PomMapping(_usePixelDisplacementRange,numSteps,viewDirUV,mask, samplerMask,xz,lod);

            uvs.uvxzOffset = final.yz;
            height += node_Blend.y * final.x;
            NdotV  += node_Blend.y * viewDirTS.z;
        }

        UNITY_BRANCH if (node_Blend.z >= 0.001)
        {
            float3 viewDirTS = float3(V.x,V.y,abs(V.z));
            float3 viewDirUV = normalize(float3(viewDirTS.xy * uvSpaceScale, viewDirTS.z));
            float  unitAngle = saturate(FastACosPos(viewDirUV.z) * INV_HALF_PI);
            int    numSteps  = (int)lerp(_usePixelDisplacementSteps - 10, _usePixelDisplacementSteps, unitAngle);
            final    = PomMapping(_usePixelDisplacementRange,numSteps,viewDirUV,mask, samplerMask,xy,lod);

            uvs.uvxyOffset = final.yz;
            height += node_Blend.z * final.x;
            NdotV  += node_Blend.z * viewDirTS.z;
        }
    }
#else
    {
        float3 viewDirTS = TransformWorldToTangent(V, tbn);
        NdotV = viewDirTS.z;
        float3 viewDirUV    = normalize(float3(viewDirTS.xy * uvSpaceScale, viewDirTS.z));
        float  unitAngle    = saturate(FastACosPos(viewDirUV.z) * INV_HALF_PI);
        int    numSteps     = (int)lerp(_usePixelDisplacementSteps - 10, _usePixelDisplacementSteps, unitAngle);
        final = PomMapping(_usePixelDisplacementRange,numSteps,viewDirUV,mask, samplerMask,uv,lod);
        height = final.x;
        uvs.uvOffset = final.yz;
    }
#endif

    float verticalDisplacement = maxHeight - height * maxHeight;
    verticalDisplacement /= max(NdotV, 0.0001);
    return verticalDisplacement;
#else
    return 0;
#endif
}

float4 CustomSampleAlbedoGrad(TEXTURE2D_PARAM(textureName, samplerName), float2 uv, float2 dxuv, float2 dyuv, float4 remapScale,float2 xz,float2 xy,float2 zy,float3 Node_Blend,float4 st,float4 size,float3x3 tbn,float3 V,float worldScale,TEXTURE2D_PARAM(mask, samplerMask))
{
    UVS uvoffsets;
    InitializeUVS(uvoffsets);
    float depthOffset;
    depthOffset = ApplyPerPixelDisplacement(tbn, V, uv,xz,xy,zy,worldScale,size,st, mask, samplerMask,Node_Blend,uvoffsets);
    uv += uvoffsets.uvOffset;
    xz += uvoffsets.uvxzOffset;
    xy += uvoffsets.uvxyOffset;
    zy += uvoffsets.uvzyOffset;

#ifdef _TRIPLANAR
    float4 albedo = SAMPLE_TEXTURE2D_GRAD(textureName, samplerName, zy, ddx(zy), ddy(zy)) * Node_Blend.x;
    albedo += SAMPLE_TEXTURE2D_GRAD(textureName, samplerName, xz, ddx(xz), ddy(xz)) * Node_Blend.y;
    albedo += SAMPLE_TEXTURE2D_GRAD(textureName, samplerName, xy, ddx(xy), ddy(xy)) * Node_Blend.z;
#else
    float4 albedo = SAMPLE_TEXTURE2D_GRAD(textureName, samplerName, uv, dxuv, dyuv);
#endif
    albedo.rgb *=remapScale.xyz;
    return albedo;
}

float4 CustomSampleAlbedoGradNoTile(TEXTURE2D_PARAM(textureName, samplerName), float2 uv, float2 dxuv, float2 dyuv, float4 remapScale,float2 xz,float2 xy,float2 zy,float3 Node_Blend,float4 st,float4 size,float3x3 tbn,float3 V,float worldScale,TEXTURE2D_PARAM(mask, samplerMask))
{
    UVS uvoffsets;
    InitializeUVS(uvoffsets);
    float depthOffset;
    depthOffset = ApplyPerPixelDisplacement(tbn, V, uv,xz,xy,zy,worldScale,size,st, mask, samplerMask,Node_Blend,uvoffsets);
    uv += uvoffsets.uvOffset;
    xz += uvoffsets.uvxzOffset;
    xy += uvoffsets.uvxyOffset;
    zy += uvoffsets.uvzyOffset;

#ifdef _TRIPLANAR
    float4 albedo = texNoTile(textureName, samplerName, zy) * Node_Blend.x;
    albedo += texNoTile(textureName, samplerName, xz) * Node_Blend.y;
    albedo += texNoTile(textureName, samplerName, xy) * Node_Blend.z;
#else
    float4 albedo = texNoTile(textureName, samplerName, uv);
#endif
    albedo.rgb *=remapScale.xyz;
    return albedo;
}

float3 CustomSampleNormalGrad(TEXTURE2D_PARAM(textureName, samplerName), float2 uv, float2 dxuv, float2 dyuv, float scale,float2 xz,float2 xy,float2 zy,float3 Node_Blend,float4 st,float4 size,float3x3 tbn,float3 V,float worldScale,TEXTURE2D_PARAM(mask, samplerMask))
{
    UVS uvoffsets;
    InitializeUVS(uvoffsets);
    float depthOffset;
    depthOffset = ApplyPerPixelDisplacement(tbn, V, uv,xz,xy,zy,worldScale,size,st, mask, samplerMask,Node_Blend,uvoffsets);
    uv += uvoffsets.uvOffset;
    xz += uvoffsets.uvxzOffset;
    xy += uvoffsets.uvxyOffset;
    zy += uvoffsets.uvzyOffset;

#ifdef _TRIPLANAR
    float4 nrm = SAMPLE_TEXTURE2D_GRAD(textureName, samplerName, zy, ddx(zy), ddy(zy)) * Node_Blend.x;
    nrm += SAMPLE_TEXTURE2D_GRAD(textureName, samplerName, xz, ddx(xz), ddy(xz)) * Node_Blend.y;
    nrm += SAMPLE_TEXTURE2D_GRAD(textureName, samplerName, xy, ddx(xy), ddy(xy)) * Node_Blend.z;
#else
    float4 nrm = SAMPLE_TEXTURE2D_GRAD(textureName, samplerName, uv, dxuv, dyuv);
#endif
#ifdef SURFACE_GRADIENT
    #ifdef UNITY_NO_DXT5nm
        return float3(UnpackDerivativeNormalRGB(nrm, scale), 0);
    #else
        return float3(UnpackDerivativeNormalRGorAG(nrm, scale), 0);
    #endif
#else
    #ifdef UNITY_NO_DXT5nm
        return UnpackNormalRGB(nrm, scale);
    #else
        return UnpackNormalmapRGorAG(nrm, scale);
    #endif
#endif
}

float3 CustomSampleNormalGradNoTile(TEXTURE2D_PARAM(textureName, samplerName), float2 uv, float2 dxuv, float2 dyuv, float scale,float2 xz,float2 xy,float2 zy,float3 Node_Blend,float4 st,float4 size,float3x3 tbn,float3 V,float worldScale,TEXTURE2D_PARAM(mask, samplerMask))
{
    UVS uvoffsets;
    InitializeUVS(uvoffsets);
    float depthOffset;
    depthOffset = ApplyPerPixelDisplacement(tbn, V, uv,xz,xy,zy,worldScale,size,st, mask, samplerMask,Node_Blend,uvoffsets);
    uv += uvoffsets.uvOffset;
    xz += uvoffsets.uvxzOffset;
    xy += uvoffsets.uvxyOffset;
    zy += uvoffsets.uvzyOffset;

#ifdef _TRIPLANAR
    float4 nrm = texNoTile(textureName, samplerName, zy) * Node_Blend.x;
    nrm += texNoTile(textureName, samplerName, xz) * Node_Blend.y;
    nrm += texNoTile(textureName, samplerName, xy) * Node_Blend.z;
#else
    float4 nrm = texNoTile(textureName, samplerName, uv);
#endif
#ifdef SURFACE_GRADIENT
    #ifdef UNITY_NO_DXT5nm
        return float3(UnpackDerivativeNormalRGB(nrm, scale), 0);
    #else
        return float3(UnpackDerivativeNormalRGorAG(nrm, scale), 0);
    #endif
#else
    #ifdef UNITY_NO_DXT5nm
        return UnpackNormalRGB(nrm, scale);
    #else
        return UnpackNormalmapRGorAG(nrm, scale);
    #endif
#endif
}

float4 CustomSampleMaskGrad(TEXTURE2D_PARAM(textureName, samplerName), float2 uv, float2 dxuv, float2 dyuv, float scale,float2 xz,float2 xy,float2 zy,float3 Node_Blend,float4 st,float4 size,float3x3 tbn,float3 V,float worldScale,TEXTURE2D_PARAM(mask, samplerMask))
{
    UVS uvoffsets;
    InitializeUVS(uvoffsets);
    float depthOffset;
    depthOffset = ApplyPerPixelDisplacement(tbn, V, uv,xz,xy,zy,worldScale,size,st, mask, samplerMask,Node_Blend,uvoffsets);
    uv += uvoffsets.uvOffset;
    xz += uvoffsets.uvxzOffset;
    xy += uvoffsets.uvxyOffset;
    zy += uvoffsets.uvzyOffset;

#ifdef _TRIPLANAR
    float4 msk = SAMPLE_TEXTURE2D_GRAD(textureName, samplerName, zy, ddx(zy), ddy(zy)) * Node_Blend.x;
    msk += SAMPLE_TEXTURE2D_GRAD(textureName, samplerName, xz, ddx(xz), ddy(xz)) * Node_Blend.y;
    msk += SAMPLE_TEXTURE2D_GRAD(textureName, samplerName, xy, ddx(xy), ddy(xy)) * Node_Blend.z;
#else
    float4 msk = SAMPLE_TEXTURE2D_GRAD(textureName, samplerName, uv, dxuv, dyuv);
#endif
    return msk;
}

float4 CustomSampleMaskGradNoTile(TEXTURE2D_PARAM(textureName, samplerName), float2 uv, float2 dxuv, float2 dyuv, float4 remapScale,float2 xz,float2 xy,float2 zy,float3 Node_Blend,float4 st,float4 size,float3x3 tbn,float3 V,float worldScale,TEXTURE2D_PARAM(mask, samplerMask))
{
    UVS uvoffsets;
    InitializeUVS(uvoffsets);
    float depthOffset;
    depthOffset = ApplyPerPixelDisplacement(tbn, V, uv,xz,xy,zy,worldScale,size,st, mask, samplerMask,Node_Blend,uvoffsets);
    uv += uvoffsets.uvOffset;
    xz += uvoffsets.uvxzOffset;
    xy += uvoffsets.uvxyOffset;
    zy += uvoffsets.uvzyOffset;

#ifdef _TRIPLANAR
    float4 msk = texNoTile(textureName, samplerName, zy) * Node_Blend.x;
    msk += texNoTile(textureName, samplerName, xz) * Node_Blend.y;
    msk += texNoTile(textureName, samplerName, xy) * Node_Blend.z;
#else
    float4 msk = texNoTile(textureName, samplerName, uv);
#endif
    return msk;
}


void CustomTerrainSplatBlend(float2 controlUV, float2 splatBaseUV,float2 xz,float2 xy,float2 zy,float3 Node_Blend,float worldScale, float3x3 tbn,float3 V,inout TerrainLitSurfaceData surfaceData)
{
    // TODO: triplanar
    // TODO: POM

    float4 albedo[_LAYER_COUNT];
    float3 normal[_LAYER_COUNT];
    float4 masks[_LAYER_COUNT];

    xz /= worldScale;
    xy /= worldScale;
    zy /= worldScale;                                                       

    
#define SampleAlbedo(i) CustomSampleAlbedoGrad(_Splat##i,sampler_Splat0, splatuv, splatdxuv, splatdyuv,_DiffuseRemapScale##i,xz*_Splat##i##_ST.xy,xy*_Splat##i##_ST.xy,zy*_Splat##i##_ST.xy,Node_Blend,_Splat##i##_ST,_Splat##i##_TexelSize,tbn,V,worldScale,_Mask##i,sampler_Splat0)
#define SampleAlbedoNoTile(i) CustomSampleAlbedoGradNoTile(_Splat##i,sampler_Splat0, splatuv, splatdxuv, splatdyuv,_DiffuseRemapScale##i,xz*_Splat##i##_ST.xy,xy*_Splat##i##_ST.xy,zy*_Splat##i##_ST.xy,Node_Blend,_Splat##i##_ST,_Splat##i##_TexelSize,tbn,V,worldScale,_Mask##i,sampler_Splat0)

#ifdef _NORMALMAP
    #define SampleNormal(i) CustomSampleNormalGrad(_Normal##i, sampler_Splat0, splatuv, splatdxuv, splatdyuv, _NormalScale##i,xz*_Splat##i##_ST.xy,xy*_Splat##i##_ST.xy,zy*_Splat##i##_ST.xy,Node_Blend,_Splat##i##_ST,_Splat##i##_TexelSize,tbn,V,worldScale,_Mask##i,sampler_Splat0)
    #define SampleNormalNoTile(i) CustomSampleNormalGradNoTile(_Normal##i, sampler_Splat0, splatuv, splatdxuv, splatdyuv, _NormalScale##i,xz*_Splat##i##_ST.xy,xy*_Splat##i##_ST.xy,zy*_Splat##i##_ST.xy,Node_Blend,_Splat##i##_ST,_Splat##i##_TexelSize,tbn,V,worldScale,_Mask##i,sampler_Splat0)
#else
    #define SampleNormal(i) float3(0, 0, 0)
    #define SampleNormalNoTile(i) float3(0, 0, 0)
#endif

#define DefaultMask(i) float4(_Metallic##i, _MaskMapRemapOffset##i.y + _MaskMapRemapScale##i.y, _MaskMapRemapOffset##i.z + 0.5 * _MaskMapRemapScale##i.z, albedo[i].a * _Smoothness##i)

#ifdef _MASKMAP
        #define CustomMaskModeMasks(i, blendMask) RemapMasks(CustomSampleMaskGrad(_Mask##i,sampler_Splat0, splatuv, splatdxuv, splatdyuv, _NormalScale##i,xz*_Splat##i##_ST.xy,xy*_Splat##i##_ST.xy,zy*_Splat##i##_ST.xy,Node_Blend,_Splat##i##_ST,_Splat##i##_TexelSize,tbn,V,worldScale,_Mask##i,sampler_Splat0), blendMask, _MaskMapRemapOffset##i, _MaskMapRemapScale##i)
        #define CustomMaskModeMasksNotile(i, blendMask) RemapMasks(CustomSampleMaskGradNoTile(_Mask##i,sampler_Splat0, splatuv, splatdxuv, splatdyuv, _NormalScale##i,xz*_Splat##i##_ST.xy,xy*_Splat##i##_ST.xy,zy*_Splat##i##_ST.xy,Node_Blend,_Splat##i##_ST,_Splat##i##_TexelSize,tbn,V,worldScale,_Mask##i,sampler_Splat0), blendMask, _MaskMapRemapOffset##i, _MaskMapRemapScale##i)
        #define SampleMasks(i, blendMask) lerp(DefaultMask(i), CustomMaskModeMasks(i, blendMask), _LayerHasMask##i)
        #define SampleMasksNotile(i, blendMask) lerp(DefaultMask(i), CustomMaskModeMasksNotile(i, blendMask), _LayerHasMask##i)
        #define NullMask(i)               float4(0, 1, _MaskMapRemapOffset##i.z, 0) // only height matters when weight is zero.
#else
    #define SampleMasks(i, blendMask) DefaultMask(i)
    #define SampleMasksNotile(i, blendMask) DefaultMask(i)
    #define NullMask(i)               float4(0, 1, 0, 0)
#endif

#define SampleResults(i, mask)                                                                              \
UNITY_BRANCH if (mask > 0)                                                                                  \
{                                                                                                           \
    float2 splatuv = splatBaseUV * _Splat##i##_ST.xy + _Splat##i##_ST.zw;                                   \
    float2 splatdxuv = dxuv * _Splat##i##_ST.x;                                                             \
    float2 splatdyuv = dyuv * _Splat##i##_ST.y;                                                             \
    albedo[i] = SampleAlbedo(i);                                                                            \
    normal[i] = SampleNormal(i);                                                                            \
    masks[i] = SampleMasks(i, mask);                                                                        \
}                                                                                                           \
else                                                                                                        \
{                                                                                                           \
    albedo[i] = float4(0, 0, 0, 0);                                                                         \
    normal[i] = float3(0, 0, 0);                                                                            \
    masks[i] = NullMask(i);                                                                                 \
}                                                                                                           
                                                                                                            
#define SampleResultsNotile(i, mask)                                                                        \
UNITY_BRANCH if (mask > 0)                                                                                  \
{                                                                                                           \
    float2 splatuv = splatBaseUV * _Splat##i##_ST.xy + _Splat##i##_ST.zw;                                   \
    float2 splatdxuv = dxuv * _Splat##i##_ST.x;                                                             \
    float2 splatdyuv = dyuv * _Splat##i##_ST.y;                                                             \
    albedo[i]= SampleAlbedoNoTile(i);                                                                       \
    normal[i] = SampleNormalNoTile(i);                                                                      \
    masks[i] = SampleMasksNotile(i, mask);                                                                  \
}                                                                                                           \
else                                                                                                        \
{                                                                                                           \
    albedo[i] = float4(0, 0, 0, 0);                                                                         \
    normal[i] = float3(0, 0, 0);                                                                            \
    masks[i] = NullMask(i);                                                                                 \
}                                                                                                           

    // Derivatives are not available for ray tracing for now
#if defined(SHADER_STAGE_RAY_TRACING)
    float2 dxuv = 0;
    float2 dyuv = 0;
#else
    float2 dxuv = ddx(splatBaseUV);
    float2 dyuv = ddy(splatBaseUV);
#endif

    float2 blendUV0 = (controlUV.xy * (_Control0_TexelSize.zw - 1.0f) + 0.5f) * _Control0_TexelSize.xy;
    float4 blendMasks0 = SAMPLE_TEXTURE2D(_Control0, sampler_Control0, blendUV0);
    #ifdef _TERRAIN_8_LAYERS
        float2 blendUV1 = (controlUV.xy * (_Control1_TexelSize.zw - 1.0f) + 0.5f) * _Control1_TexelSize.xy;
        float4 blendMasks1 = SAMPLE_TEXTURE2D(_Control1, sampler_Control0, blendUV1);
    #else
        float4 blendMasks1 = float4(0, 0, 0, 0);
    #endif

#ifdef _NOTILE
    SampleResultsNotile(0, blendMasks0.x);
    SampleResultsNotile(1, blendMasks0.y);
    SampleResultsNotile(2, blendMasks0.z);
    SampleResultsNotile(3, blendMasks0.w);
    #ifdef _TERRAIN_8_LAYERS
        SampleResultsNotile(4, blendMasks1.x);
        SampleResultsNotile(5, blendMasks1.y);
        SampleResultsNotile(6, blendMasks1.z);
        SampleResultsNotile(7, blendMasks1.w);
    #endif
#else
    SampleResults(0, blendMasks0.x);
    SampleResults(1, blendMasks0.y);
    SampleResults(2, blendMasks0.z);
    SampleResults(3, blendMasks0.w);
    #ifdef _TERRAIN_8_LAYERS
        SampleResults(4, blendMasks1.x);
        SampleResults(5, blendMasks1.y);
        SampleResults(6, blendMasks1.z);
        SampleResults(7, blendMasks1.w);
    #endif
#endif

#undef SampleNormal
#undef SampleMasks
#undef SampleResults
#undef SampleNormalNoTile
#undef SampleMasksNotile
#undef SampleResultsNotile

    float weights[_LAYER_COUNT];
    ZERO_INITIALIZE_ARRAY(float, weights, _LAYER_COUNT);

    #ifdef _MASKMAP
        #if defined(_TERRAIN_BLEND_HEIGHT)
            // Modify blendMask to take into account the height of the layer. Higher height should be more visible.
            float maxHeight = masks[0].z;
            maxHeight = max(maxHeight, masks[1].z);
            maxHeight = max(maxHeight, masks[2].z);
            maxHeight = max(maxHeight, masks[3].z);
            #ifdef _TERRAIN_8_LAYERS
                maxHeight = max(maxHeight, masks[4].z);
                maxHeight = max(maxHeight, masks[5].z);
                maxHeight = max(maxHeight, masks[6].z);
                maxHeight = max(maxHeight, masks[7].z);
            #endif

            // Make sure that transition is not zero otherwise the next computation will be wrong.
            // The epsilon here also has to be bigger than the epsilon in the next computation.

            // sheep yang
            //float transition = max(_HeightTransition, 1e-5);
            float transition = max(1-_HeightTransition, 1e-5);

            // The goal here is to have all but the highest layer at negative heights, then we add the transition so that if the next highest layer is near transition it will have a positive value.
            // Then we clamp this to zero and normalize everything so that highest layer has a value of 1.
            float4 weightedHeights0 = { masks[0].z, masks[1].z, masks[2].z, masks[3].z };
            weightedHeights0 = weightedHeights0 - maxHeight.xxxx;
            // We need to add an epsilon here for active layers (hence the blendMask again) so that at least a layer shows up if everything's too low.
            weightedHeights0 = (max(0, weightedHeights0 + transition) + 1e-6) * blendMasks0;

            #ifdef _TERRAIN_8_LAYERS
                float4 weightedHeights1 = { masks[4].z, masks[5].z, masks[6].z, masks[7].z };
                weightedHeights1 = weightedHeights1 - maxHeight.xxxx;
                weightedHeights1 = (max(0, weightedHeights1 + transition) + 1e-6) * blendMasks1;
            #else
                float4 weightedHeights1 = { 0, 0, 0, 0 };
            #endif

            // Normalize
            float sumHeight = GetSumHeight(weightedHeights0, weightedHeights1);
            blendMasks0 = weightedHeights0 / sumHeight.xxxx;
            #ifdef _TERRAIN_8_LAYERS
                blendMasks1 = weightedHeights1 / sumHeight.xxxx;
            #endif
        #elif defined(_TERRAIN_BLEND_DENSITY)
            // Denser layers are more visible.
            float4 opacityAsDensity0 = saturate((float4(albedo[0].a, albedo[1].a, albedo[2].a, albedo[3].a) - (float4(1.0, 1.0, 1.0, 1.0) - blendMasks0)) * 20.0); // 20.0 is the number of steps in inputAlphaMask (Density mask. We decided 20 empirically)
            opacityAsDensity0 += 0.001f * blendMasks0;      // if all weights are zero, default to what the blend mask says
            float4 useOpacityAsDensityParam0 = { _DiffuseRemapScale0.w, _DiffuseRemapScale1.w, _DiffuseRemapScale2.w, _DiffuseRemapScale3.w }; // 1 is off
            blendMasks0 = lerp(opacityAsDensity0, blendMasks0, useOpacityAsDensityParam0);
            #ifdef _TERRAIN_8_LAYERS
                float4 opacityAsDensity1 = saturate((float4(albedo[4].a, albedo[5].a, albedo[6].a, albedo[7].a) - (float4(1.0, 1.0, 1.0, 1.0) - blendMasks1)) * 20.0); // 20.0 is the number of steps in inputAlphaMask (Density mask. We decided 20 empirically)
                opacityAsDensity1 += 0.001f * blendMasks1;  // if all weights are zero, default to what the blend mask says
                float4 useOpacityAsDensityParam1 = { _DiffuseRemapScale4.w, _DiffuseRemapScale5.w, _DiffuseRemapScale6.w, _DiffuseRemapScale7.w };
                blendMasks1 = lerp(opacityAsDensity1, blendMasks1, useOpacityAsDensityParam1);
            #endif

            // Normalize
            float sumHeight = GetSumHeight(blendMasks0, blendMasks1);
            blendMasks0 = blendMasks0 / sumHeight.xxxx;
            #ifdef _TERRAIN_8_LAYERS
                blendMasks1 = blendMasks1 / sumHeight.xxxx;
            #endif
        #endif // if _TERRAIN_BLEND_HEIGHT
    #endif // if _MASKMAP

    weights[0] = blendMasks0.x;
    weights[1] = blendMasks0.y;
    weights[2] = blendMasks0.z;
    weights[3] = blendMasks0.w;
    #ifdef _TERRAIN_8_LAYERS
        weights[4] = blendMasks1.x;
        weights[5] = blendMasks1.y;
        weights[6] = blendMasks1.z;
        weights[7] = blendMasks1.w;
    #endif

    surfaceData.albedo = 0;
    surfaceData.normalData = 0;
    float3 outMasks = 0;
    UNITY_UNROLL for (int i = 0; i < _LAYER_COUNT; ++i)
    {
        surfaceData.albedo += albedo[i].rgb * weights[i];
        surfaceData.normalData += normal[i].rgb * weights[i]; // no need to normalize
        outMasks += masks[i].xyw * weights[i];
    }
    surfaceData.smoothness = outMasks.z;
    surfaceData.metallic = outMasks.x;
    surfaceData.ao = outMasks.y;
}

//sheep yang
void CustomTerrainLitShade(float2 uv,float2 xz,float2 xy,float2 zy,float3 Node_Blend, float worldScale, float3x3 tbn,float3 V,inout TerrainLitSurfaceData surfaceData)
{
    CustomTerrainSplatBlend(uv, uv, xz, xy, zy, Node_Blend,worldScale,tbn,V,surfaceData);
}

void TerrainLitDebug(float2 uv, inout float3 baseColor)
{
#ifdef DEBUG_DISPLAY
    if (_DebugMipMapModeTerrainTexture == DEBUGMIPMAPMODETERRAINTEXTURE_CONTROL)
        baseColor = GetTextureDataDebug(_DebugMipMapMode, uv, _Control0, _Control0_TexelSize, _Control0_MipInfo, baseColor);
    else if (_DebugMipMapModeTerrainTexture == DEBUGMIPMAPMODETERRAINTEXTURE_LAYER0)
        baseColor = GetTextureDataDebug(_DebugMipMapMode, uv * _Splat0_ST.xy + _Splat0_ST.zw, _Splat0, _Splat0_TexelSize, _Splat0_MipInfo, baseColor);
    else if (_DebugMipMapModeTerrainTexture == DEBUGMIPMAPMODETERRAINTEXTURE_LAYER1)
        baseColor = GetTextureDataDebug(_DebugMipMapMode, uv * _Splat1_ST.xy + _Splat1_ST.zw, _Splat1, _Splat1_TexelSize, _Splat1_MipInfo, baseColor);
    else if (_DebugMipMapModeTerrainTexture == DEBUGMIPMAPMODETERRAINTEXTURE_LAYER2)
        baseColor = GetTextureDataDebug(_DebugMipMapMode, uv * _Splat2_ST.xy + _Splat2_ST.zw, _Splat2, _Splat2_TexelSize, _Splat2_MipInfo, baseColor);
    else if (_DebugMipMapModeTerrainTexture == DEBUGMIPMAPMODETERRAINTEXTURE_LAYER3)
        baseColor = GetTextureDataDebug(_DebugMipMapMode, uv * _Splat3_ST.xy + _Splat3_ST.zw, _Splat3, _Splat3_TexelSize, _Splat3_MipInfo, baseColor);
    #ifdef _TERRAIN_8_LAYERS
        else if (_DebugMipMapModeTerrainTexture == DEBUGMIPMAPMODETERRAINTEXTURE_LAYER4)
            baseColor = GetTextureDataDebug(_DebugMipMapMode, uv * _Splat4_ST.xy + _Splat4_ST.zw, _Splat4, _Splat4_TexelSize, _Splat4_MipInfo, baseColor);
        else if (_DebugMipMapModeTerrainTexture == DEBUGMIPMAPMODETERRAINTEXTURE_LAYER5)
            baseColor = GetTextureDataDebug(_DebugMipMapMode, uv * _Splat5_ST.xy + _Splat5_ST.zw, _Splat5, _Splat5_TexelSize, _Splat5_MipInfo, baseColor);
        else if (_DebugMipMapModeTerrainTexture == DEBUGMIPMAPMODETERRAINTEXTURE_LAYER6)
            baseColor = GetTextureDataDebug(_DebugMipMapMode, uv * _Splat6_ST.xy + _Splat6_ST.zw, _Splat6, _Splat6_TexelSize, _Splat6_MipInfo, baseColor);
        else if (_DebugMipMapModeTerrainTexture == DEBUGMIPMAPMODETERRAINTEXTURE_LAYER7)
            baseColor = GetTextureDataDebug(_DebugMipMapMode, uv * _Splat7_ST.xy + _Splat7_ST.zw, _Splat7, _Splat7_TexelSize, _Splat7_MipInfo, baseColor);
    #endif
#endif
}
