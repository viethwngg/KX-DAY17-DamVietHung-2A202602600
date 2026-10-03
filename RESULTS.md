# Kết quả bài lab Day 17

## Phạm vi đã hoàn thành

Đã triển khai cấu hình, sáu provider, hai agent, `User.md`, compact memory, benchmark hai bộ dữ liệu và 33 test. Bonus gồm entity extraction theo field, conflict handling cho correction, chống lưu câu hỏi/câu giả định thành fact, tránh trùng fact và tách biệt dữ liệu giữa các user.

## Cách chạy và đo

```powershell
.\.venv\Scripts\python.exe src/benchmark.py --json-output benchmark_results.json
.\.venv\Scripts\python.exe -m pytest src/test_agents.py -v
```

Kết quả dưới đây được đo ở chế độ offline deterministic, ngưỡng compact 1.400 token, giữ 6 message gần nhất. Benchmark tạo profile rỗng riêng cho từng suite, đánh giá sau từng conversation, hỏi mỗi câu recall ở một thread mới. File [benchmark_results.json](benchmark_results.json) lưu các giá trị số để kiểm tra lại.

- **Agent tokens only**: tổng token đầu ra ước lượng, gồm lượt hội thoại và lượt recall. Không tính đầu vào; không có LLM riêng để trích fact hay tóm tắt.
- **Prompt tokens processed**: tổng ngữ cảnh xử lý qua tất cả lượt, gồm system prompt, lịch sử; Advanced thêm profile và summary. Có overhead 4 token/message. Đây là estimator thống nhất, không phải tokenizer hay hóa đơn của provider.
- **Cross-session recall**: trung bình tỷ lệ chuỗi kỳ vọng có mặt trong câu trả lời, không phân biệt hoa thường, chuẩn hóa Unicode. Không dùng `expected_contains` để sinh phản hồi.
- **Response quality**: proxy `0.8 × recall + 0.1 × (độ dài ≤ 800 ký tự) + 0.1 × (câu trả lời không rỗng)`. Vì gắn với recall, đây không phải phép chấm độc lập về văn phong hay suy luận. Baseline có 20% nhờ câu trả lời ngắn, không rỗng dù thiếu facts.
- **Memory growth**: chênh lệch kích thước UTF-8 của các `User.md` trước/sau suite, không bao gồm RAM của lịch sử hoặc summary. Baseline không tạo file profile.
- **Compactions**: số lần lịch sử cũ được gộp vào summary. Summary và recent messages nằm trong RAM theo thread, không tồn tại qua lần khởi động mới.

## Số liệu đo được

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 4275 | 36965 | 0.0% | 20.0% | 0 | 0 |
| Advanced | 4362 | 45848 | 100.0% | 100.0% | 306 | 0 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 760 | 27607 | 0.0% | 20.0% | 0 | 0 |
| Advanced | 789 | 16929 | 100.0% | 100.0% | 213 | 3 |

## Phân tích

**Vì sao Advanced nhớ tốt hơn?** Baseline dùng cùng bộ trích fact và cùng chính sách trả lời, nhưng chỉ đọc các message trong thread đang hỏi. Thread recall mới không chứa facts cũ nên trả lời chưa biết. Advanced đọc `User.md` bất kể thread; test còn dựng lại một instance mới để chứng minh hồ sơ tồn tại trên đĩa. Recall 100% chỉ áp dụng cho các câu hỏi về facts trong hai dataset này.

**Hội thoại ngắn:** Advanced xử lý thêm 24.0% prompt tokens so với Baseline. Không có compaction nào trong suite standard. Chi phí tăng đến từ việc đưa profile vào mỗi prompt, trong khi lịch sử từng conversation vẫn ngắn. Token đầu ra cũng có thể tăng vì trả lời đúng nhiều facts hơn và duy trì style cá nhân. Vì vậy thêm persistent memory không tự động giảm chi phí.

**Hội thoại dài:** Advanced compact 3 lần, giảm prompt tokens từ 27,607 xuống 16,929, tương đương 38.7%. Baseline tiếp tục mang toàn bộ lịch sử và đầu ra cũ vào mỗi lượt. Advanced thay các message cũ bằng summary có giới hạn, giữ phần gần nhất đầy đủ và đọc facts từ profile. Lợi ích chính nằm ở ngữ cảnh lặp lại, không phải số token đầu ra. Test còn so Advanced bật/tắt compact để cô lập tác động của cơ chế này.

**Correction và nhiễu:** Nơi ở của user standard đổi từ Đà Nẵng sang Huế; nghề đổi từ backend engineer sang MLOps engineer. User stress đổi Huế sang Đà Nẵng. Các scalar field được upsert nên không giữ hai giá trị mâu thuẫn. Hà Nội đi họp và product manager trong câu đùa không ghi đè hồ sơ. Câu hỏi có chứa tên hoặc thành phố không được dùng như một phát biểu fact.

**Memory growth:** Hai profile cuối cùng lần lượt là 306 và 213 bytes. Ghi theo field và loại trùng khiến việc nhắc lại cùng fact không làm file tăng mãi. Các field như style và interests vẫn có thể tăng khi có nhiều preference mới; chưa có memory decay hoặc chính sách hết hạn. Phần ghi chú Markdown do người dùng tự thêm được giữ nguyên.

## Giới hạn và hướng cải tiến

- Extraction dựa trên pattern tiếng Việt và các loại nghề phổ biến; chưa phải NER tổng quát, chưa có confidence score bằng model. Các diễn đạt ngoài pattern có thể bị bỏ sót. Các sở thích mới được gộp; chưa xử lý đầy đủ việc người dùng rút lại sở thích cũ.
- Summary giữ facts cùng một số trích đoạn theo chủ đề; có thể mất chi tiết hoặc mốc số. Bốn tin trong stress test là nội dung input, không phải dữ kiện tin tức đã được hệ thống xác minh. Benchmark hiện chỉ chấm recall profile, chưa chấm chất lượng nhớ các tin hay abstraction sau compact.
- Ngưỡng compact là ngưỡng mềm: các message được giữ nguyên gần nhất có thể tự lớn hơn ngưỡng. Summary bị giới hạn ký tự; profile hiện tại được ưu tiên nếu summary cũ mâu thuẫn.
- History và summary không lưu qua lần khởi động lại. Các dictionary thread chưa có TTL/eviction. Ghi profile dùng atomic replace nhưng chưa có khóa để giải quyết nhiều tiến trình cùng sửa một user.
- Live đã kiểm tra việc khởi tạo cả sáu integration mà không gửi request, cùng đường gọi model và metadata token bằng model giả. Chưa benchmark API thật; chưa triển khai tool agent LangGraph hoặc judge LLM. Live vẫn dùng memory layer deterministic.

## Kiểm chứng

33 test pass, bao gồm read/write/edit và ghi chú Markdown, đường dẫn user an toàn, tách user/thread, compact nhiều lần, giảm prompt load, recall sau khi dựng lại agent, correction, chống nhiễu và câu hỏi, không tăng file khi lặp facts, gộp summary, benchmark tuần tự, token accounting, cấu hình environment, sáu provider, live context/usage và ép offline không tạo model thật.
