#ifndef TERRAINLIT_FEEDBACKRENDERER_INCLUDED
#define TERRAINLIT_FEEDBACKRENDERER_INCLUDED

sampler2D _MainTex;
float4 _MainTex_TexelSize;
float4 _VTFeedbackParam;

#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
 #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Packing.hlsl"
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/TextureStack.hlsl"
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/NormalSurfaceGradient.hlsl"
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Color.hlsl"
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Texture.hlsl"
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/UnityInstancing.hlsl"
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/EntityLighting.hlsl"
#include "Packages/com.unity.shadergraph/ShaderGraphLibrary/ShaderVariables.hlsl"
#include "Packages/com.unity.shadergraph/ShaderGraphLibrary/ShaderVariablesFunctions.hlsl"
#include "Packages/com.unity.shadergraph/ShaderGraphLibrary/Functions.hlsl"

struct feed_attr
{
    float4 vertex : POSITION;
    float2 texcoord : TEXCOORD0;
};

struct feed_v2f
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

feed_v2f VertFeedbackRenderer(feed_attr v)
{
    feed_v2f o;
    //VertexPositionInputs Attributes = GetVertexPositionInputs(v.vertex.xyz);
    o.pos = TransformObjectToHClip(v.vertex.xyz);
    o.uv = v.texcoord;
    
    return o;
}

float4 FragFeedbackRenderer(feed_v2f i) : SV_Target
{
    float2 page = floor(i.uv * _VTFeedbackParam.x);

    float2 uv = i.uv * _VTFeedbackParam.y;
    float2 dx = ddx(uv);
    float2 dy = ddy(uv);
    int mip = clamp(int(0.5 * log2(max(dot(dx, dx), dot(dy, dy))) + 0.5), 0, _VTFeedbackParam.z);

    return float4(page / 255.0, mip / 255.0, 1);
}


float4 GetMaxFeedback(float2 uv, int count)
{
    float4 col = float4(1, 1, 1, 1);
    for (int y = 0; y < count; y++)
    {
        for (int x = 0; x < count; x++)
        {
            float4 col1 = tex2D(_MainTex, uv + float2(_MainTex_TexelSize.x * x, _MainTex_TexelSize.y * y));
            col = lerp(col, col1, step(col1.b, col.b));
        }
    }
    return col;
}

#endif