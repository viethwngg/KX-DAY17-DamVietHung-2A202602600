# Bước 8 — Phân tích kết quả benchmark

**Sinh viên:** Đàm Việt Hùng  
**MSSV:** 2A202602600

Phần này trả lời bốn yêu cầu của bước 8 trong [Guide.md](Guide.md), dựa trên kết quả chạy hai bộ dữ liệu gốc ở chế độ offline. Cấu hình compact: ngưỡng 1.400 token, giữ 6 message gần nhất.

## Kết quả đo

| Bộ benchmark | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Standard | Baseline | 4.275 | 36.965 | 0% | 20% | 0 | 0 |
| Standard | Advanced | 4.362 | 45.848 | 100% | 100% | 306 | 0 |
| Long-context | Baseline | 760 | 27.607 | 0% | 20% | 0 | 0 |
| Long-context | Advanced | 789 | 16.929 | 100% | 100% | 213 | 3 |

Token offline được ước lượng từ số ký tự; các tổng token bao gồm cả lượt hội thoại và lượt recall. Quality là điểm heuristic dựa trên recall, độ ngắn và câu trả lời không rỗng, chưa phải đánh giá bằng judge LLM. Số liệu gốc được lưu trong [benchmark_results.json](benchmark_results.json).

## 1. Vì sao Advanced có recall tốt hơn Baseline?

Baseline chỉ lưu lịch sử theo `thread_id`. Khi benchmark hỏi recall ở thread mới, agent không có các thông tin từ thread trước nên trả lời chưa biết. Advanced lưu những fact ổn định vào `User.md`, rồi đọc hồ sơ này trong mỗi thread mới. Vì vậy Advanced vẫn nhớ tên, nghề nghiệp, nơi ở, đồ uống và style trả lời dù lịch sử thread cũ không được đưa vào prompt.

Các field nơi ở và nghề nghiệp được cập nhật bằng giá trị mới nhất. Trong bộ standard, nơi ở đổi từ Đà Nẵng sang Huế và nghề đổi từ backend engineer sang MLOps engineer. Trong bộ stress, nơi ở đổi từ Huế sang Đà Nẵng. Câu đùa về product manager và chuyến đi họp Hà Nội không ghi đè những fact này. Nhờ đó Advanced đạt recall 100%, còn Baseline đạt 0% trên các câu hỏi chéo phiên của hai bộ dữ liệu.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Ở Standard Benchmark, Advanced xử lý 45.848 prompt tokens, cao hơn khoảng **24,0%** so với 36.965 của Baseline. Các thread chưa đủ dài để kích hoạt compact, nên Advanced vẫn giữ lịch sử gần đây và còn đưa thêm `User.md` vào mỗi prompt. Chi phí đọc hồ sơ chưa được bù bằng lợi ích nén lịch sử.

Agent tokens only cũng tăng từ 4.275 lên 4.362 vì Advanced trả lời được nhiều fact hơn và duy trì style của người dùng. Persistent memory cải thiện recall nhưng có thêm chi phí lưu trữ và xử lý ngữ cảnh; không bảo đảm tiết kiệm token trong mọi tình huống.

## 3. Vì sao compact giúp Advanced có lợi thế ở hội thoại dài?

Baseline đưa toàn bộ lịch sử vào mỗi lượt, khiến các đoạn văn dài được xử lý lặp lại nhiều lần. Advanced chuyển message cũ thành summary có giới hạn, giữ 6 message gần nhất đầy đủ và sử dụng profile làm nguồn fact hiện tại. Do đó lượng ngữ cảnh mang theo không tiếp tục tăng như lịch sử nguyên văn.

Trong Long-Context Stress Benchmark, compact xảy ra **3 lần**. Prompt tokens giảm từ 27.607 xuống 16.929, tương đương **38,7%**, trong khi recall profile vẫn đạt 100%. Token đầu ra của Advanced vẫn cao hơn một chút: 789 so với 760. Điều này cho thấy compact chủ yếu tối ưu **prompt tokens processed**, thay vì làm câu trả lời ngắn hơn.

Summary có thể mất chi tiết khi nén. Kết quả recall ở đây chỉ kiểm tra các fact trong profile; chưa chứng minh agent nhớ chính xác toàn bộ nội dung tin tức sau compact. Ngưỡng compact cũng là ngưỡng mềm vì những message gần nhất được giữ nguyên có thể tự vượt ngưỡng.

## 4. File memory tăng trưởng ra sao và có rủi ro gì?

Profile cuối của bộ standard có kích thước **306 bytes**, còn profile của bộ stress có **213 bytes**. Baseline không có file profile nên memory growth bằng 0. Chỉ số này đo kích thước `User.md` trên đĩa, không tính lịch sử và summary trong RAM.

Hồ sơ lưu theo field, thay thế fact cũ và loại bỏ giá trị trùng nên việc nhắc lại cùng một thông tin không làm file phình mãi. Tuy nhiên style, interests và ghi chú vẫn có thể tăng khi người dùng bổ sung thông tin mới. File càng lớn thì mỗi lượt đọc profile càng tốn ngữ cảnh.

Các rủi ro còn lại gồm trích sai fact, giữ thông tin đã lỗi thời, mất chi tiết khi compact và xung đột khi nhiều tiến trình cùng sửa hồ sơ. Bản triển khai đã lọc câu hỏi/câu giả định, xử lý correction và ghi file bằng atomic replace, nhưng chưa có memory decay, cơ chế rút lại preference hay khóa ghi giữa các tiến trình. Có thể cải thiện bằng thời điểm cập nhật cho từng fact, kiểm tra độ tin cậy trước khi lưu và chính sách hết hạn cho thông tin ít được sử dụng.
