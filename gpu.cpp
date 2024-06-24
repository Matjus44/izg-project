/*!
 * @file
 * @brief This file contains implementation of gpu
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */

#include <student/gpu.hpp>
#include <algorithm> // pre std::min a std::max

typedef struct Triangle {
  OutVertex points[3];
} primitive;




//! Funkce pro nastavení hloubky framebufferu.
//! \param fbo Framebuffer, na který se má hloubka aplikovat.
//! \param depth Hloubka, na kterou se má framebuffer čistit.
void setFramebufferDepth(Framebuffer* fbo, float depth) 
{
    // Výpočet počtu pixelů v framebufferu
    uint32_t pixelsCount = fbo->width * fbo->height;
    // Přetypování ukazatele na pole bajtů
    uint8_t* pixelData = static_cast<uint8_t*>(fbo->depth.data);
    // Převod hloubky na bajty
    uint32_t depthBytes = *reinterpret_cast<uint32_t*>(&depth);
    // Nastavení všech pixelů hloubkového bufferu na zadanou hloubku
    for (uint32_t i = 0; i < pixelsCount; ++i) 
    {
        // Kopírování bajtů hloubky na správné místo v paměti hloubkového bufferu
        for (uint32_t j = 0; j < fbo->depth.bytesPerPixel; ++j) 
        {
            pixelData[i * fbo->depth.bytesPerPixel + j] = (depthBytes >> (j * 8)) & 0xFF;
        }
    }
}

void setFramebufferColor(Framebuffer* fbo, const glm::vec4& color) 
{
    // Výpočet počtu pixelů v framebufferu
    uint32_t pixelsCount = fbo->width * fbo->height;
    // Přetypování ukazatele na pole bajtů
    uint8_t* pixelData = static_cast<uint8_t*>(fbo->color.data);
    // Převod barvy z rozsahu [0,1] na rozsah [0,255] a nastavení všech pixelů
    uint8_t r = static_cast<uint8_t>(color.r * 255);
    uint8_t g = static_cast<uint8_t>(color.g * 255);
    uint8_t b = static_cast<uint8_t>(color.b * 255);
    uint8_t a = static_cast<uint8_t>(color.a * 255);
    for (uint32_t i = 0; i < pixelsCount; ++i) 
    {
        pixelData[i * fbo->color.channels + Image::RED] = r;
        pixelData[i * fbo->color.channels + Image::GREEN] = g;
        pixelData[i * fbo->color.channels + Image::BLUE] = b;
        pixelData[i * fbo->color.channels + Image::ALPHA] = a;
    }
}

void clear(GPUMemory& mem, const ClearCommand& cmd) 
{
    Framebuffer* fbo = mem.framebuffers + mem.activatedFramebuffer;

    if (fbo) 
    {
        if (cmd.clearColor) 
        {
            if (fbo->color.data) 
            {
                setFramebufferColor(fbo, cmd.color);
            }
        }
        if (cmd.clearDepth) 
        {
            if (fbo->depth.data) 
            {
                setFramebufferDepth(fbo, cmd.depth);
            }
        }
    }
}

void assembleVertex(const GPUMemory& mem, const VertexArray& va, uint32_t vertexIndex, InVertex& outVertex) 
{
    for (int attribIndex = 0; attribIndex < maxAttributes; ++attribIndex) 
    {
        const VertexAttrib& attrib = va.vertexAttrib[attribIndex];
        if (attrib.type != AttributeType::EMPTY && attrib.bufferID != -1) 
        {
            const Buffer& buffer = mem.buffers[attrib.bufferID];
            const char* bufferData = reinterpret_cast<const char*>(buffer.data);
            uint64_t byteOffset = attrib.offset + vertexIndex * attrib.stride;
            
            switch (attrib.type) 
            {
                case AttributeType::FLOAT:
                    outVertex.attributes[attribIndex].v1 = *reinterpret_cast<const float*>(bufferData + byteOffset);
                    break;
                case AttributeType::VEC2:
                    outVertex.attributes[attribIndex].v2 = *reinterpret_cast<const glm::vec2*>(bufferData + byteOffset);
                    break;
                case AttributeType::VEC3:
                    outVertex.attributes[attribIndex].v3 = *reinterpret_cast<const glm::vec3*>(bufferData + byteOffset);
                    break;
                case AttributeType::VEC4:
                    outVertex.attributes[attribIndex].v4 = *reinterpret_cast<const glm::vec4*>(bufferData + byteOffset);
                    break;
                case AttributeType::UINT:
                    outVertex.attributes[attribIndex].u1 = *reinterpret_cast<const uint32_t*>(bufferData + byteOffset);
                    break;
                case AttributeType::UVEC2:
                    outVertex.attributes[attribIndex].u2 = *reinterpret_cast<const glm::uvec2*>(bufferData + byteOffset);
                    break;
                case AttributeType::UVEC3:
                    outVertex.attributes[attribIndex].u3 = *reinterpret_cast<const glm::uvec3*>(bufferData + byteOffset);
                    break;
                case AttributeType::UVEC4:
                    outVertex.attributes[attribIndex].u4 = *reinterpret_cast<const glm::uvec4*>(bufferData + byteOffset);
                    break;
            }
        }
    }
    outVertex.gl_VertexID = vertexIndex;
}

void runPrimitiveAssembly(primitive& prim, const VertexArray& va, uint32_t t, const Program& program, const GPUMemory& mem) 
{
    for (int v = 0; v < 3; ++v) 
    {
        InVertex inVertex;
        assembleVertex(mem, va, t + v, inVertex);
        ShaderInterface si;
        si.uniforms = mem.uniforms;
        si.textures = mem.textures;
        si.gl_DrawID = mem.gl_DrawID;
        program.vertexShader(prim.points[v], inVertex, si);
    }
}

void runPerspectiveDivision(primitive& prim) 
{
    // for (int i = 0; i < 3; ++i) 
    // {
    //     prim.points[i].gl_Position /= prim.points[i].gl_Position.w;
    // }

    for (int i = 0; i < 3; ++i) 
    {
        prim.points[i].gl_Position.x /= prim.points[i].gl_Position.w;
        prim.points[i].gl_Position.y /= prim.points[i].gl_Position.w;
        prim.points[i].gl_Position.z /= prim.points[i].gl_Position.w;
        // prim.points[i].gl_Position.w zostáva nedotknuté
    }
}

void runViewportTransformation(primitive& prim, int width, int height) 
{
    for (int i = 0; i < 3; ++i) 
    {
        prim.points[i].gl_Position.x = (prim.points[i].gl_Position.x * 0.5f + 0.5f) * width;
        prim.points[i].gl_Position.y = (prim.points[i].gl_Position.y * 0.5f + 0.5f) * height;
    }
}

bool isBackfaceCullingEnabled(const DrawCommand& cmd)
{
    return cmd.backfaceCulling;
}

bool isBackface(const primitive& prim) 
{
    glm::vec3 a = glm::vec3(prim.points[1].gl_Position) - glm::vec3(prim.points[0].gl_Position);
    glm::vec3 b = glm::vec3(prim.points[2].gl_Position) - glm::vec3(prim.points[0].gl_Position);
    glm::vec3 normal = glm::cross(a, b);
    return normal.z < 0;
}

glm::vec3 computeBarycentrics(const primitive& prim, const glm::vec2& pixel) 
{
    glm::vec3 barycentrics;
    glm::vec4 p0 = prim.points[0].gl_Position;
    glm::vec4 p1 = prim.points[1].gl_Position;
    glm::vec4 p2 = prim.points[2].gl_Position;

    float area = ((p1.y - p2.y) * (p0.x - p2.x) + (p2.x - p1.x) * (p0.y - p2.y));
    barycentrics.x = ((p1.y - p2.y) * (pixel.x - p2.x) + (p2.x - p1.x) * (pixel.y - p2.y)) / area;
    barycentrics.y = ((p2.y - p0.y) * (pixel.x - p2.x) + (p0.x - p2.x) * (pixel.y - p2.y)) / area;
    barycentrics.z = 1.0f - barycentrics.x - barycentrics.y;

    return barycentrics;
}

bool isInsideTriangle(const glm::vec3& barycentrics) 
{
    return barycentrics.x >= 0 && barycentrics.y >= 0 && barycentrics.z >= 0;
}

void createFragment(InFragment& frag, const primitive& prim, const glm::vec3& barycentrics, const glm::vec2& pixel, const Program& program) 
{
    // 2D Barycentric coordinates
    float lambda0_2D = barycentrics.x;
    float lambda1_2D = barycentrics.y;
    float lambda2_2D = barycentrics.z;

    // Homogeneous coordinates
    float h0 = prim.points[0].gl_Position.w;
    float h1 = prim.points[1].gl_Position.w;
    float h2 = prim.points[2].gl_Position.w;

    // Perspective-correct barycentric coordinates
    float lambda0 = lambda0_2D / h0;
    float lambda1 = lambda1_2D / h1;
    float lambda2 = lambda2_2D / h2;

    // Sum of perspective-correct barycentric coordinates
    float w_sum = lambda0 + lambda1 + lambda2;

    // Normalize perspective-correct barycentric coordinates
    lambda0 /= w_sum;
    lambda1 /= w_sum;
    lambda2 /= w_sum;

    // Interpolate attributes-
    for (int i = 0; i < maxAttributes; i++) 
    {
        if (program.vs2fs[i] == AttributeType::EMPTY) continue;

        if (program.vs2fs[i] == AttributeType::FLOAT) 
        {
            frag.attributes[i].v1 = prim.points[0].attributes[i].v1 * lambda0 +
                                    prim.points[1].attributes[i].v1 * lambda1 +
                                    prim.points[2].attributes[i].v1 * lambda2;
        } 
        else if (program.vs2fs[i] == AttributeType::VEC2) 
        {
            frag.attributes[i].v2 = prim.points[0].attributes[i].v2 * lambda0 +
                                    prim.points[1].attributes[i].v2 * lambda1 +
                                    prim.points[2].attributes[i].v2 * lambda2;
        } 
        else if (program.vs2fs[i] == AttributeType::VEC3) 
        {
            frag.attributes[i].v3 = prim.points[0].attributes[i].v3 * lambda0 +
                                    prim.points[1].attributes[i].v3 * lambda1 +
                                    prim.points[2].attributes[i].v3 * lambda2;
        } 
        else if (program.vs2fs[i] == AttributeType::VEC4) 
        {
            frag.attributes[i].v4 = prim.points[0].attributes[i].v4 * lambda0 +
                                    prim.points[1].attributes[i].v4 * lambda1 +
                                    prim.points[2].attributes[i].v4 * lambda2;
        } 
        else if (program.vs2fs[i] == AttributeType::UINT ||
                 program.vs2fs[i] == AttributeType::UVEC2 ||
                 program.vs2fs[i] == AttributeType::UVEC3 ||
                 program.vs2fs[i] == AttributeType::UVEC4) 
        {
            frag.attributes[i] = prim.points[0].attributes[i]; // Provoking vertex
        }
    }

    // Calculate fragment depth without perspective correction.
    float z0 = prim.points[0].gl_Position.z;
    float z1 = prim.points[1].gl_Position.z;
    float z2 = prim.points[2].gl_Position.z;
    frag.gl_FragCoord = glm::vec4(pixel, z0 * lambda0_2D + z1 * lambda1_2D + z2 * lambda2_2D, 1.0f);
}



void writeFragment(Framebuffer* fbo, int x, int y, const OutFragment& frag, float depth) 
{
    if (x < 0 || x >= fbo->width || y < 0 || y >= fbo->height) return;

    uint32_t index = y * fbo->width + x;
    float* depthData = static_cast<float*>(fbo->depth.data);

    if (depth < depthData[index]) 
    {
        depthData[index] = depth;

        uint8_t* pixelData = static_cast<uint8_t*>(fbo->color.data);

        glm::vec4 fragColor = glm::clamp(frag.gl_FragColor, 0.0f, 1.0f);
        float alpha = fragColor.a;

        glm::vec3 framebufferColor = glm::vec3(
            pixelData[index * fbo->color.channels + Image::RED] / 255.0f,
            pixelData[index * fbo->color.channels + Image::GREEN] / 255.0f,
            pixelData[index * fbo->color.channels + Image::BLUE] / 255.0f
        );

        framebufferColor = framebufferColor * (1.0f - alpha) + glm::vec3(fragColor) * alpha;

        pixelData[index * fbo->color.channels + Image::RED] = static_cast<uint8_t>(framebufferColor.r * 255);
        pixelData[index * fbo->color.channels + Image::GREEN] = static_cast<uint8_t>(framebufferColor.g * 255);
        pixelData[index * fbo->color.channels + Image::BLUE] = static_cast<uint8_t>(framebufferColor.b * 255);
        pixelData[index * fbo->color.channels + Image::ALPHA] = static_cast<uint8_t>(1.0f * 255);
    }
}


void perFragmentOperations(Framebuffer* fbo, const OutFragment& outFragment, float fragDepth, int x, int y) 
{
    if (outFragment.discard) return;

    writeFragment(fbo, x, y, outFragment, fragDepth);
}

void rasterizeTriangle(Framebuffer* fbo, const primitive& prim, const Program& program, const GPUMemory& mem) 
{
    int minX = std::max(0.0f, std::min({prim.points[0].gl_Position.x, prim.points[1].gl_Position.x, prim.points[2].gl_Position.x}));
    int maxX = std::min(static_cast<float>(fbo->width - 1), std::max({prim.points[0].gl_Position.x, prim.points[1].gl_Position.x, prim.points[2].gl_Position.x}));
    int minY = std::max(0.0f, std::min({prim.points[0].gl_Position.y, prim.points[1].gl_Position.y, prim.points[2].gl_Position.y}));
    int maxY = std::min(static_cast<float>(fbo->height - 1), std::max({prim.points[0].gl_Position.y, prim.points[1].gl_Position.y, prim.points[2].gl_Position.y}));

    for (int x = minX; x <= maxX; ++x) 
    {
        for (int y = minY; y <= maxY; ++y) 
        {
            glm::vec2 pixel(x + 0.5f, y + 0.5f);
            glm::vec3 barycentrics = computeBarycentrics(prim, pixel);
            if (isInsideTriangle(barycentrics)) 
            {
                InFragment inFragment;
                createFragment(inFragment, prim, barycentrics, pixel, program);
                OutFragment outFragment;
                ShaderInterface si;
                si.uniforms = mem.uniforms;
                si.textures = mem.textures;
                program.fragmentShader(outFragment, inFragment, si);
                perFragmentOperations(fbo, outFragment, inFragment.gl_FragCoord.z, x, y);
            }
        }
    }
}


void draw(GPUMemory& mem, const DrawCommand& cmd) 
{
    Framebuffer* fbo = mem.framebuffers + mem.activatedFramebuffer;
    const Program& program = mem.programs[mem.activatedProgram];
    const VertexArray& va = mem.vertexArrays[mem.activatedVertexArray];

    for (uint32_t t = 0; t < cmd.nofVertices; t += 3) 
    {
        primitive prim;
        runPrimitiveAssembly(prim, va, t, program, mem);

        if (isBackfaceCullingEnabled(cmd) && isBackface(prim)) 
        {
            continue;
        }

        runPerspectiveDivision(prim);
        runViewportTransformation(prim, fbo->width, fbo->height);
        rasterizeTriangle(fbo, prim, program, mem);
    }
}

void drawIndexed(GPUMemory& mem, DrawCommand cmd, const VertexArray& va) 
{
    Buffer indexBuffer = mem.buffers[va.indexBufferID];
    const char* indices = reinterpret_cast<const char*>(indexBuffer.data) + va.indexOffset;
    Program& program = mem.programs[mem.activatedProgram];

    for (uint32_t n = 0; n < cmd.nofVertices; ++n) 
    {
        uint32_t index = 0;
        if (va.indexType == IndexType::UINT8) 
        {
            index = reinterpret_cast<const uint8_t*>(indices)[n];
        } 
        else if (va.indexType == IndexType::UINT16) 
        {
            index = reinterpret_cast<const uint16_t*>(indices)[n];
        } 
        else if (va.indexType == IndexType::UINT32) 
        {
            index = reinterpret_cast<const uint32_t*>(indices)[n];
        }

        InVertex inVertex;
        inVertex.gl_VertexID = index;
        OutVertex outVertex;
        ShaderInterface si;

        assembleVertex(mem, va, index, inVertex);

        si.uniforms = mem.uniforms;
        si.textures = mem.textures;
        si.gl_DrawID = mem.gl_DrawID;

        program.vertexShader(outVertex, inVertex, si);
    }
}



//! [izg_enqueue]
void izg_enqueue(GPUMemory&mem,CommandBuffer const&cb)
{
    mem.gl_DrawID = 0;

    for (uint32_t i = 0; i < cb.nofCommands; ++i) 
    {
        CommandType type = cb.commands[i].type;
        CommandData data = cb.commands[i].data;

        if (type == CommandType::BIND_FRAMEBUFFER) 
        {
            uint32_t framebufferId = data.bindFramebufferCommand.id;
            mem.activatedFramebuffer = framebufferId;
        }
        else if(type == CommandType::BIND_PROGRAM)
        {
            uint32_t framebufferId = data.bindProgramCommand.id;
            mem.activatedProgram = framebufferId;
        }
        else if(type == CommandType::BIND_VERTEXARRAY)
        {
            uint32_t framebufferId = data.bindVertexArrayCommand.id;
            mem.activatedVertexArray = framebufferId;
        }
        else if (type == CommandType::DRAW) 
        {
            const VertexArray& va = mem.vertexArrays[mem.activatedVertexArray];
            if (va.indexBufferID != -1) 
            {
                drawIndexed(mem, data.drawCommand, va);
            } 
            else 
            {
                draw(mem, data.drawCommand);
            }
            mem.gl_DrawID++;
        }
        else if (type == CommandType::SET_DRAW_ID)      
        {
            mem.gl_DrawID = data.setDrawIdCommand.id;
        }
        else if (type == CommandType::SUB_COMMAND)
        {
            if (data.subCommand.commandBuffer != nullptr) 
            {
                izg_enqueue(mem, *data.subCommand.commandBuffer); 
            }
        }
        else
        {
            clear(mem, data.clearCommand);
        }
    }
}
//! [izg_enqueue]

/**
 * @brief This function reads color from texture.
 *
 * @param texture texture
 * @param uv uv coordinates
 *
 * @return color 4 floats
 */
glm::vec4 read_texture(Texture const&texture,glm::vec2 uv){
  if(!texture.img.data)return glm::vec4(0.f);
  auto&img = texture.img;
  auto uv1 = glm::fract(glm::fract(uv)+1.f);
  auto uv2 = uv1*glm::vec2(texture.width-1,texture.height-1)+0.5f;
  auto pix = glm::uvec2(uv2);
  return texelFetch(texture,pix);
}

/**
 * @brief This function reads color from texture with clamping on the borders.
 *
 * @param texture texture
 * @param uv uv coordinates
 *
 * @return color 4 floats
 */
glm::vec4 read_textureClamp(Texture const&texture,glm::vec2 uv){
  if(!texture.img.data)return glm::vec4(0.f);
  auto&img = texture.img;
  auto uv1 = glm::clamp(uv,0.f,1.f);
  auto uv2 = uv1*glm::vec2(texture.width-1,texture.height-1)+0.5f;
  auto pix = glm::uvec2(uv2);
  return texelFetch(texture,pix);
}

/**
 * @brief This function fetches color from texture.
 *
 * @param texture texture
 * @param pix integer coorinates
 *
 * @return color 4 floats
 */
glm::vec4 texelFetch(Texture const&texture,glm::uvec2 pix){
  auto&img = texture.img;
  glm::vec4 color = glm::vec4(0.f,0.f,0.f,1.f);
  if(pix.x>=texture.width || pix.y >=texture.height)return color;
  if(img.format == Image::UINT8){
    auto colorPtr = (uint8_t*)getPixel(img,pix.x,pix.y);
    for(uint32_t c=0;c<img.channels;++c)
      color[c] = colorPtr[img.channelTypes[c]]/255.f;
  }
  if(texture.img.format == Image::FLOAT32){
    auto colorPtr = (float*)getPixel(img,pix.x,pix.y);
    for(uint32_t c=0;c<img.channels;++c)
      color[c] = colorPtr[img.channelTypes[c]];
  }
  return color;
}

