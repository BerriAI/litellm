#!/usr/bin/env python3
"""
Gemini 3 Pro Image Preview 모델 테스트 스크립트

이 스크립트는 gemini-3-pro-image-preview 모델의 다음 기능을 테스트합니다:
1. Chat Completion API를 통한 이미지 생성
2. Image Generation API를 통한 이미지 생성
3. imageSize 파라미터 (1K, 2K, 4K) 지원
4. aspectRatio 파라미터 지원
5. Google AI Studio와 Vertex AI 모두 지원

사용 방법:
    export GEMINI_API_KEY="your-api-key-here"
    # Vertex AI 사용 시:
    export VERTEX_PROJECT="your-project-id"
    export VERTEX_LOCATION="us-central1"
    
    poetry run python test_gemini_3_pro_image.py
"""

import litellm
import os
import base64
from datetime import datetime

# 디버그 로깅 활성화
os.environ['LITELLM_LOG'] = 'DEBUG'
litellm.set_verbose = True

# 출력 디렉토리 생성
output_dir = "gemini_3_pro_test_outputs"
os.makedirs(output_dir, exist_ok=True)


def save_image_from_base64(base64_data, filename):
    """base64 데이터를 이미지 파일로 저장"""
    try:
        image_bytes = base64.b64decode(base64_data)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        print(f"✅ 이미지 저장 완료: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ 이미지 저장 실패 ({filename}): {e}")
        return None


def test_completion_api():
    """Chat Completion API 테스트"""
    print("\n" + "="*80)
    print("Chat Completion API 테스트 (Google AI Studio)")
    print("="*80 + "\n")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("테스트를 건너뜁니다.")
        return
    
    try:
        # 테스트 1: 기본 이미지 생성 (aspectRatio만)
        print("\n1️⃣ Chat Completion - 기본 이미지 생성 (aspectRatio만)")
        print("-" * 80)
        response = litellm.completion(
            model="gemini/gemini-3-pro-image-preview",
            messages=[{
                "role": "user",
                "content": "Generate a beautiful landscape of Mount Fuji at sunrise with cherry blossoms"
            }],
            api_key=api_key,
            imageConfig={
                "aspectRatio": "16:9"
            },
            response_modalities=["Image"],
        )
        
        print(f"✅ 생성 완료!")
        if hasattr(response.choices[0].message, 'images') and response.choices[0].message.images:
            for i, img_obj in enumerate(response.choices[0].message.images):
                if img_obj.image_url and img_obj.image_url.url:
                    base64_data = img_obj.image_url.url.split(",")[1]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_image_from_base64(base64_data, f"completion_basic_{timestamp}_{i}.png")
        
        # 테스트 2: imageSize 파라미터 (4K)
        print("\n2️⃣ Chat Completion - 4K 이미지 생성 (imageSize=4K)")
        print("-" * 80)
        response_4k = litellm.completion(
            model="gemini/gemini-3-pro-image-preview",
            messages=[{
                "role": "user",
                "content": "Generate a futuristic Tokyo skyline at night with neon lights"
            }],
            api_key=api_key,
            imageConfig={
                "aspectRatio": "16:9",
                "imageSize": "4K"
            },
            response_modalities=["Image"],
        )
        
        print(f"✅ 4K 이미지 생성 완료!")
        if hasattr(response_4k.choices[0].message, 'images') and response_4k.choices[0].message.images:
            for i, img_obj in enumerate(response_4k.choices[0].message.images):
                if img_obj.image_url and img_obj.image_url.url:
                    base64_data = img_obj.image_url.url.split(",")[1]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_image_from_base64(base64_data, f"completion_4k_{timestamp}_{i}.png")
        
        # Thinking 확인 (있다면)
        print("\n🧠 Thinking 데이터 확인:")
        for i, choice in enumerate(response_4k.choices):
            if hasattr(choice.message, 'thinking_blocks') and choice.message.thinking_blocks:
                print(f"   Choice {i}: {len(choice.message.thinking_blocks)} thinking blocks 발견")
            else:
                print(f"   Choice {i}: Thinking blocks 없음")
        
    except litellm.exceptions.BadRequestError as e:
        print(f"❌ BadRequestError: {e}")
        print(f"   에러 메시지: {e.message}")
    except Exception as e:
        print(f"❌ 예기치 않은 오류: {e}")


def test_image_generation_api():
    """Image Generation API 테스트"""
    print("\n" + "="*80)
    print("Image Generation API 테스트 (Google AI Studio)")
    print("="*80 + "\n")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("테스트를 건너뜁니다.")
        return
    
    try:
        # 테스트 1: 기본 이미지 생성
        print("\n1️⃣ Image Generation - 기본 이미지 생성")
        print("-" * 80)
        response = litellm.image_generation(
            model="gemini/gemini-3-pro-image-preview",
            prompt="Generate a cute cat wearing a tiny wizard hat",
            api_key=api_key,
            imageConfig={
                "aspectRatio": "1:1"
            }
        )
        
        print(f"✅ 생성 완료!")
        for i, img_obj in enumerate(response.data):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_image_from_base64(img_obj.b64_json, f"image_gen_basic_{timestamp}_{i}.png")
        
        # 테스트 2: 2K 이미지 생성
        print("\n2️⃣ Image Generation - 2K 이미지 생성 (imageSize=2K)")
        print("-" * 80)
        response_2k = litellm.image_generation(
            model="gemini/gemini-3-pro-image-preview",
            prompt="Generate a spaceship landing on a desert planet",
            api_key=api_key,
            imageConfig={
                "aspectRatio": "16:9",
                "imageSize": "2K"
            }
        )
        
        print(f"✅ 2K 이미지 생성 완료!")
        for i, img_obj in enumerate(response_2k.data):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_image_from_base64(img_obj.b64_json, f"image_gen_2k_{timestamp}_{i}.png")
        
    except litellm.exceptions.BadRequestError as e:
        print(f"❌ BadRequestError: {e}")
        print(f"   에러 메시지: {e.message}")
    except Exception as e:
        print(f"❌ 예기치 않은 오류: {e}")


def test_vertex_ai():
    """Vertex AI 테스트"""
    print("\n" + "="*80)
    print("Vertex AI 테스트")
    print("="*80 + "\n")
    
    project = os.getenv("VERTEX_PROJECT")
    location = os.getenv("VERTEX_LOCATION", "us-central1")
    
    if not project:
        print("⚠️  VERTEX_PROJECT 환경 변수가 설정되지 않았습니다.")
        print("Vertex AI 테스트를 건너뜁니다.")
        return
    
    try:
        print("\n1️⃣ Vertex AI Completion - 1K 이미지 생성")
        print("-" * 80)
        response = litellm.completion(
            model="vertex_ai/gemini-3-pro-image-preview",
            messages=[{
                "role": "user",
                "content": "Generate a vibrant abstract painting for a modern art gallery"
            }],
            vertex_project=project,
            vertex_location=location,
            imageConfig={
                "aspectRatio": "4:3",
                "imageSize": "1K"
            },
            response_modalities=["Image"],
        )
        
        print(f"✅ Vertex AI 이미지 생성 완료!")
        if hasattr(response.choices[0].message, 'images') and response.choices[0].message.images:
            for i, img_obj in enumerate(response.choices[0].message.images):
                if img_obj.image_url and img_obj.image_url.url:
                    base64_data = img_obj.image_url.url.split(",")[1]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_image_from_base64(base64_data, f"vertex_ai_{timestamp}_{i}.png")
        
    except litellm.exceptions.BadRequestError as e:
        print(f"❌ BadRequestError: {e}")
        print(f"   에러 메시지: {e.message}")
    except Exception as e:
        print(f"❌ 예기치 않은 오류: {e}")


def main():
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + " "*20 + "Gemini 3 Pro Image Preview 모델 테스트" + " "*20 + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    print("\n이 테스트는 gemini-3-pro-image-preview 모델의 기능을 검증합니다:")
    print("  • Chat Completion API를 통한 이미지 생성")
    print("  • Image Generation API를 통한 이미지 생성")
    print("  • imageSize 파라미터 (1K, 2K, 4K) 지원")
    print("  • aspectRatio 파라미터 지원")
    print("  • Thinking 기능 (자동으로 활성화됨)")
    print("="*80 + "\n")
    
    # 테스트 실행
    test_completion_api()
    test_image_generation_api()
    test_vertex_ai()
    
    print("\n" + "="*80)
    print("테스트 완료!")
    print(f"생성된 이미지는 '{output_dir}/' 디렉토리에 저장되었습니다.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
