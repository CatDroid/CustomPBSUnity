Shader "CustomPBS/CustomPBR"
{
    Properties
    {
        // Color和MainTex(ALbedo) 控制漫反射项的颜色 
        _Color ("Color", Color) = (1, 1, 1, 1)
        _MainTex ("Albedo", 2D) = "white" {}
        // 光滑度/粗糙度 roughness = (1-smoothness)  roughness^2 计算BRDF中的高光项中的几何函数和法线分布函数
        _Glossiness ("Smoothness", Range(0.0,1.0)) = 0.5 
        // 高光颜色
        _SpecColor ("Specular", Color) = (0.2, 0.2, 0.2)
        // 高光反射图(金属工作流中 albedo漫反射颜色决定非金属材质的颜色, specular高光反射颜色决定金属材质的颜色)
        _SpecGlossMap ("Specular(RGB) Smoothness(A)", 2D) = "white" {} 
        // 法线 凹凸程度
        _BumpScale ("Bump Scale", FLoat) = 1.0 
        _BumpMap ("Normal Map", 2D) = "bump" {}
        // 自发光 默认不发光
        _EmissionColor ("Emission Color", Color) = (0, 0, 0)
        _EmissionMap ("Emission", 2D) = "white" {} 
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 300

        Pass
        {
            CGPROGRAM

            // 这个cgprogram 需要指令比较多 会超出Shader Target2.0对指令数目的限制 
            #pragma target 3.0 

            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"
            #include "Lighting.cginc"   // 灯光 _LightColor0
            #include "AutoLight.cginc"  // 阴影 
            

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                float3 normal : NORMAL;
                float4 tangent : TANGENT;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                float4 TtoW0: TEXCOORD1;
                float4 TtoW1: TEXCOORD2;
                float4 TtoW2: TEXCOORD3;
                SHADOW_COORDS(4)            // AutoLight.cginc 阴影 
                UNITY_FOG_COORDS(5)
            };


            float4 _Color;
            sampler2D _MainTex;
            float4 _MainTex_ST;

            float _Glossiness;
            //float3 _SpecColor;
            sampler2D _SpecGlossMap;
            float4 _SpecGlossMap_ST;

            float _BumpScale;
            sampler2D _BumpMap;
            float4 _BumpMap_ST;

            float3 _EmissionColor;
            sampler2D _EmissionMap;
            float4 _EmissionMap_ST;


            v2f vert (appdata v)
            {
                v2f o;
                UNITY_INITIALIZE_OUTPUT(v2f, o); // HLSLSupport.cginc 初始化输出变量 


                o.pos = UnityObjectToClipPos(v.vertex);  // UnityCG.cginc 
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);       // UnityCG.cginc 

                float3 worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                float3 worldNormal = UnityObjectToWorldNormal(v.normal);
                float3 worldTangents = UnityObjectToWorldDir(v.tangent.xyz);
                float3 worldBinormal = cross(worldNormal, worldTangents) * v.tangent.w ; // 切线的w可以是-1或者1 纹理y轴有关系 
                // 一般法线作为z轴  切线作为x轴  所以不管左右手坐标系 都是法线cross切线 

                // 在ps把凹凸map中的法线 转换到 世界坐标系 
                o.TtoW0 = float4(worldNormal.x, worldBinormal.x, worldTangents.x, worldPos.x);
                o.TtoW1 = float4(worldNormal.y, worldBinormal.y, worldTangents.y, worldPos.y);
                o.TtoW2 = float4(worldNormal.z, worldBinormal.z, worldTangents.z, worldPos.z);
                

                TRANSFER_SHADOW(o); // 为了接收阴影 

                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

 

            // 高光反射强度 max(specular.r, specular.g, specular.b); 
            inline half _SpecularStrength(half3 specular)
            {
                return max( max(specular.r, specular.g), specular.b);
            }

            inline half3 _EnergyConservationBetweenDiffuseAndSpecular(half3 albedo, half3 specColor, out half oneMinusReflectivity)
            {
                oneMinusReflectivity = 1 - _SpecularStrength(specColor);
                // UnityStandardCore.cginc 
                // 对于没有定义 UNITY_CONSERVE_ENERGY 都直接返回 albedo 因为下面这个公式完全可预先计算好保存到Albedo(MainTex)
                // 定义了 UNITY_CONSERV_ENERGY 返回 return albedo * (half3(1,1,1) - specColor);  1.0减去高光反射
                // 按这样说 ???  MainTex.rgb * _Color + SpecGLossMap.rgb*_SpecColor = 1.0 ??? 
                return albedo ;
            }

            // 对 MainTex.rgb 采样 并且乘以参数_Color.rgb 并且和 DetailAlbedoMap.rgb 以 DetailMask.a 做混合 得到慢反射颜色
            // texcoords.xy   _MainTex
            // texcoords.wz   _DetailAlbedoMap
            inline half3 Albedo(float2 texcoords) // UnityStandardInput.cginc 
            {
                half3 albedo = tex2D(_MainTex, texcoords.xy).rgb * _Color.rgb;
                // UnityStandardCore.cginc 会判断 _DETAIL宏定义、
                // 采集 _MainTex.rgb(uv.xy) 和  _DetailAlbedoMap.rgb(uv.zw) 根据 DetailMask.a; 做某种插值 
                return albedo ;
            }

            // 法线xy保存在RG 还是 AG  UnityStandardUtils.cginc 
            inline half3 _UnpackScaleNormalRGorAG(half4 packednormal, half bumpScale)
            {
                // #if defined(UNITY_NO_DXT5nm)  // 没有使用DXT5nm压缩 纹理保存法线的xyz 
                half3 normal = packednormal.xyz * 2 - 1;
                normal.xy *= bumpScale;
                return normal; 
                // #else
                    // This do the trick  DXT5nm W存放x Y存放y  R5-G6-B5-A8 并且没有存放z 
                    // packednormal.x *= packednormal.w;
                    // half3 normal;
                    // normal.xy = (packednormal.xy * 2 - 1);
                    // normal.xy *= bumpScale;
                    // normal.z = sqrt(1.0 - saturate(dot(normal.xy, normal.xy)));
                // #endif     
            }

            inline half3 _UnpackScaleNormal(half4 packednormal, half bumpScale)
            {
                return _UnpackScaleNormalRGorAG(packednormal, bumpScale);
            }
 
            // 从 BumpMap 和 _DetailNormapMap 中获取切线空间的法线
            // texcoords.xy  BumpMap 
            // texcoords.zw  _DetailNormapMap   这个跟 MainTex 和 DetalAlbedoMap 类似 
            inline half3 NormalInTangentSpace(float2 texcoords)
            {
                // _BumpScale 缩放法线
                half3 normalTangent = _UnpackScaleNormal(tex2D (_BumpMap, texcoords.xy), _BumpScale); 

                //  #if _DETAIL && defined(UNITY_ENABLE_DETAIL_NORMALMAP)
                //  unity 会在 BumpMap.xyz 和 BumpScaleMap.xyz 中 根据DetailMask.a  做线性插值

                return normalTangent;
            }

            // 考虑 逐顶点 归一化法线
            half3 NormalizePerVertexNormal(float3 n)
            {
                #if (SHADER_TARGET < 30) 
                    return normalize(n);
                #else 
                    return n;
                #endif 
            }

             // 考虑 逐法线 归一化法线
            float3 NormalizePerPixelNormal(float3 n )
            {
                #if (SHADER_TARGET < 30)
                    return n;
                #else 
                    // return normalize((float3)n); // takes float to avoid overflow  ???
                    return normalize(n);
                #endif 
            }

            // 逐像素的世界法线 
            inline float3 PerPixelWorldNormal(float2 uv, float4 TtoW0, float4 TtoW1, float4 TtoW2)
            {
                // #ifdef _NORMALMAP 
                    // Unity会考虑是在ps还是vs做归一化法线
                    // Unity会考虑重新正交化 

                half3 normalTangent = NormalInTangentSpace(uv);
         
                float3 normalWorld = float3 (
                        dot(TtoW0.xyz , normalTangent), 
                        dot(TtoW1.xyz , normalTangent),
                        dot(TtoW2.xyz , normalTangent) );
                normalWorld = NormalizePerPixelNormal(normalWorld); 

                // #else 
                // return normalize(tangentToWorld[2].xyz); // 不使用法线贴图 直接返回矩阵中的法线 
                return normalWorld;
            }
            
            struct FragmentCommonData
            {
                half3 specColor;
                half3 diffColor;
                half perceptualRoughness;
                half oneMinusReflectivity;
                
                float3 normalWorld;
                half3 viewDir;
                half3 lightDir;
                half3 reflectDir;
                half3 halfDir;

                half atten;
            };

            inline FragmentCommonData SpecularSetup (v2f i) // (float4 i_tex)
            {

    
                half4 specGloss = tex2D(_SpecGlossMap, i.uv);

                // 光泽度  
                half glossiness = specGloss.a *  _Glossiness;

                // 粗糙度
                half perceptualRoughness = 1 - glossiness;

                // 高光反射颜色 
                half3 specColor = specGloss.rgb * _SpecColor;

                 // 计算 一减反射率, 漫反射系数（Albedo中参与漫反射的比例）
                half oneMinusReflectivity;

                // 漫反射颜色 
                half3 diffColor = _EnergyConservationBetweenDiffuseAndSpecular(Albedo(i.uv), specColor, oneMinusReflectivity);

                // 还原世界坐标
                float3 worldPos = float3(i.TtoW0.w, i.TtoW1.w, i.TtoW2.w);

                // 法线 
                float3 normalWorld = PerPixelWorldNormal(i.uv , i.TtoW0, i.TtoW1, i.TtoW2);;

                // 视线方向 
                half3 viewDir = normalize(UnityWorldSpaceViewDir(worldPos));

                // 光线方向 (点光源需要世界坐标 方向光不需要)
                half3 lightDir = normalize(UnityWorldSpaceLightDir(worldPos));

                // 反射方向 
                half3 reflectDir = reflect(-viewDir, normalWorld);

                // 半向量角
                half3 halfDir = normalize(lightDir + viewDir);

                // 光线衰减   UNITY_LIGHT_ATTENUATION已经定义了 half atten ;
                UNITY_LIGHT_ATTENUATION(atten, i, worldPos);


                FragmentCommonData o;
                UNITY_INITIALIZE_OUTPUT(FragmentCommonData, o);

                o.specColor = specColor;
                o.diffColor = diffColor;
                o.perceptualRoughness = perceptualRoughness;
                o.oneMinusReflectivity = oneMinusReflectivity;
                
                o.normalWorld = normalWorld;
                o.viewDir = viewDir;
                o.lightDir = lightDir;
                o.reflectDir = reflectDir;

                o.atten = atten;

                return o;
            }

            inline half3 _Pow5 (half3 x)
            {
                return x*x * x*x * x;
            }

            float _PerceptualRoughnessToRoughness(float perceptualRoughness)
            {
                return perceptualRoughness * perceptualRoughness;
            }

            inline half3 DisneyDiffuseTerm(half3 baseColor, half NdotL, half NdotV, half HdotL, half perceptualRoughness)
            {
                float fd90 = 0.5 + 2 *  HdotL * HdotL * perceptualRoughness;

                // Two schlick fresnel term 两个 schlick-菲涅尔项 
                half lightScatter = 1 + (fd90 - 1) * _Pow5(1 - NdotL);
                half viewScatter  = 1 + (fd90 - 1) * _Pow5(1 - NdotV);

                return  baseColor * UNITY_INV_PI * lightScatter * viewScatter;

            }

            // GGX法线分布对应的几何函数 Smith-Joint阴影遮挡函数 
            float _SmithJointGGXVisibilityTerm(float NdotL, float NdotV, float roughness)
            {
                // roughness 已经是 perceptualRoughness^2 所以这里不用 a*a 
                // 使用不用sqrt的近似公式
                float a2 = roughness;
                float lambdaV = NdotL * (NdotV * (1 - a2) + a2);
                float lambdaL = NdotV * (NdotL * (1 - a2) + a2);

                // 这里已经包含了 Gsmithjolint(l,v,n)/(n_Dot_l)(n_Dot_v) * 1/4 也就是BRDF高光反射项的分母都包含了
                return 0.5f / (lambdaV + lambdaL + 1e-5f);
            }

            // 法线分布 
            inline float _GGXTerm(float NdotH, float roughness)
            {
                float a2 = roughness * roughness;
                float d = (NdotH * a2 - NdotH) * NdotH + 1.0f ;
                return UNITY_INV_PI * a2 / (d * d  + 1e-7f); // 注意分母避免除0 
            }

            // Schlick菲涅尔近似等式  A从0度到90度  返回从F0到1
            inline half3 _FresnelTerm (half3 F0, half cosA)
            {
                half t = _Pow5 (1 - cosA); 
                return F0 + (1-F0) * t ;
            }

            // 菲涅尔插值  cosA=1(角度为0) 是F0  角度90度得到F90  角度从0到90 值从F0到F90 FO高光颜色 F90是掠射角颜色
            inline half3 _FresnelLerp (half3 F0, half3 F90, half cosA)
            {
                half t = _Pow5 (1 - cosA);   // ala Schlick interpoliation
                return lerp (F0, F90, t);
            }


            // 整合了UnityStandardCore.cginc 的 FragmentSetup + UNITY_SETUP_BRDF_INPUT/SpecularSetup
            #define MY_UNITY_SETUP_BRDF_INPUT SpecularSetup

            


            fixed4 frag (v2f i) : SV_Target
            {
            
                FragmentCommonData s = MY_UNITY_SETUP_BRDF_INPUT(i);

                half nv = saturate( dot(s.normalWorld, s.viewDir)  );
                half nl = saturate( dot(s.normalWorld, s.lightDir) );
                half nh = saturate( dot(s.normalWorld, s.halfDir)  );
                half lh = saturate( dot(s.lightDir,    s.halfDir)  );
                half lv = saturate( dot(s.lightDir,    s.viewDir)  );  // 光和视线夹角 
                
                // UNITY_BRDF_PBS

                // BRDF漫反射项 
                half3 diffuseTerm = DisneyDiffuseTerm(s.diffColor, nl, nv, lh, s.perceptualRoughness);

                // perceptualRoughness^2 做了一个非线性的映射
                // 漫反射部分用 perceptualRoughness 
                // 高光反射部分用 perceptualRoughness^2 (因为V和D公式都要做 rough^2 )
                float roughness = _PerceptualRoughnessToRoughness(s.perceptualRoughness);
                // roughness = max(roughness, 0.002); // ???


                // BRDF高光反射项 

                float V  = _SmithJointGGXVisibilityTerm(nl, nv, roughness);
                float D  = _GGXTerm(nh, roughness);
                float3 F = _FresnelTerm(s.specColor, lh); // 这是向量 rgb 可能不同 SpecColor 这里是lh 法线和视线的夹角一半
                half3 specularTerm = V * D * F; // ?? 这里不用乘 specColor


                // 自发光项 
                half3 emissionTerm = tex2D(_EmissionMap, i.uv).rgb * _EmissionColor.rgb ;

                // 间接光(只计算高光部分)
                half surfaceReduction = 1.0 / (roughness*roughness + 1.0); // 由粗糙度计算得到的'?表面衰减'修正IBL

                // 掠射角颜色 = 光滑度 + specuColor 
                half grazingTerm = saturate( (1.0 - s.perceptualRoughness) + (1.0 - s.oneMinusReflectivity)); 

                // LOD采样 从0到6 
                // 使用材质粗糙度对环境贴图进行LOD采样 因为粗糙度越大的材质,反射的环境光照应该越模糊 
                // 级数越大, 对应的纹理越小 
                half mip = s.perceptualRoughness * 6 ;
                half4 envMap = UNITY_SAMPLE_TEXCUBE_LOD(unity_SpecCube0, s.reflectDir, mip);// HLSLSupport.cginc
                half3 indirectSpecular = surfaceReduction * envMap.rgb * _FresnelLerp(s.specColor, grazingTerm, nv);


                // 渲染方程 加起来 
                half3 color = 
                    emissionTerm + 
                    UNITY_PI * (diffuseTerm + specularTerm) * (_LightColor0.rgb  * s.atten) * nl +
                    indirectSpecular;


                /*

                // gi包含了 间接光的 漫反射部分 和 高光放射部分 

                float specularTerm = V*D * UNITY_PI;

                specularTerm *= any(specColor) ? 1.0 : 0.0; // unity这里还乘以了 specColor(除了菲涅尔项)

                half grazingTerm = saturate(smoothness + (1-oneMinusReflectivity)); // 掠射角颜色 = 光滑度 + (1 - 一减反射率)  // diffColor是1.0-specularColor 
                half3 color =   diffColor * (gi.diffuse + light.color * diffuseTerm)  // ?? gi.diffuse 间接漫反射 ???   light.color * diffuseTerm是原来渲染方程的漫反射部分
                    + specularTerm * light.color * FresnelTerm (specColor, lh) //  渲染方程中的高光反射项  PI已经在上面乘了
                    + surfaceReduction * gi.specular * FresnelLerp (specColor, grazingTerm, nv); // 间接高光(菲涅尔插值) 高光颜色和掠射角颜色插值
                */ 

                UNITY_APPLY_FOG(i.fogCoord, color.rgb);

                return half4(color, 1.0);
            }
            ENDCG
        }
    }
}
