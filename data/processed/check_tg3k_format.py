import os
from PIL import Image
from collections import Counter


def check_image_mask_sizes(image_dir, mask_dir):
    # Lấy danh sách file và sắp xếp
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Thống kê tổng số lượng
    total_images = len(image_files)
    total_masks = len(mask_files)

    print(f"📊 Tổng số ảnh: {total_images}")
    print(f"📊 Tổng số mask: {total_masks}")

    if total_images != total_masks:
        print("⚠️ Số lượng ảnh và mask không khớp!")

    # Biến đếm số cặp có kích thước trùng nhau
    matching_size_count = 0
    size_distribution = Counter()

    for img_name, mask_name in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        try:
            with Image.open(img_path) as img, Image.open(mask_path) as mask:
                img_size = img.size  # (width, height)
                mask_size = mask.size

                # Ghi nhận kích thước
                size_distribution[img_size] += 1

                # Kiểm tra kích thước trùng nhau
                if img_size == mask_size:
                    matching_size_count += 1
                else:
                    print(f"❌ Kích thước không khớp: {img_name} ({img_size}) vs {mask_name} ({mask_size})")

                print(f"\n🖼  {img_name}")
                print(f"    - Image: Size: {img_size}, Mode: {img.mode}, Format: {img.format}")
                print(f"    - Mask: Size: {mask_size}, Mode: {mask.mode}, Format: {mask.format}")

        except Exception as e:
            print(f"❌ Lỗi khi đọc {img_name} hoặc {mask_name}: {e}")

    # In kết quả thống kê
    print("\n📈 Thống kê:")
    print(f"✅ Số cặp ảnh-mask có kích thước trùng nhau: {matching_size_count}/{total_images}")
    print(f"📏 Phân bố kích thước ảnh:")
    for size, count in size_distribution.items():
        print(f"    - Kích thước {size}: {count} ảnh")


image_dir = r"../raw/Thyroid Dataset/tg3k/thyroid-image"
mask_dir = r"../raw/Thyroid Dataset/tg3k/thyroid-mask"

if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
    print("❌ Không tìm thấy thư mục ảnh hoặc mask!")
else:
    check_image_mask_sizes(image_dir, mask_dir)