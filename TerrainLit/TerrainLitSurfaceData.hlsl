struct TerrainLitSurfaceData
{
    float3 albedo;
    float3 normalData;
    float smoothness;
    float metallic;
    float ao;
};

void InitializeTerrainLitSurfaceData(out TerrainLitSurfaceData surfaceData)
{
    surfaceData.albedo = 0;
    surfaceData.normalData = 0;
    surfaceData.smoothness = 0;
    surfaceData.metallic = 0;
    surfaceData.ao = 1;
}

struct UVS
{
    float2 uvxyOffset;
    float2 uvxzOffset;
    float2 uvzyOffset;
    float2 uvOffset;
};

void InitializeUVS(out UVS uvOffsets)
{
    uvOffsets.uvxyOffset = float2(0,0);
    uvOffsets.uvxzOffset = float2(0,0);
    uvOffsets.uvzyOffset = float2(0,0);
    uvOffsets.uvOffset = float2(0,0);
}