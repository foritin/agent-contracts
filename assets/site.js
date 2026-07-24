for (const button of document.querySelectorAll('.copy-code')) {
  button.addEventListener('click', async () => {
    const code = button.parentElement.querySelector('code')?.innerText ?? '';
    try { await navigator.clipboard.writeText(code); button.textContent = '已复制'; }
    catch { button.textContent = '请手动复制'; }
    setTimeout(() => { button.textContent = '复制'; }, 1300);
  });
}
const article = document.querySelector('.doc-content');
const toc = document.querySelector('.toc');
if (article && toc) {
  const headings = [...article.querySelectorAll('h2, h3, h4')];
  if (headings.length) {
    const title = document.createElement('h2'); title.textContent = '本页目录'; toc.append(title);
    const list = document.createElement('ol');
    for (const heading of headings) {
      const item = document.createElement('li');
      item.dataset.level = heading.tagName.slice(1);
      const link = document.createElement('a'); link.href = '#' + heading.id; link.textContent = heading.textContent;
      item.append(link); list.append(item);
    }
    toc.append(list);
  }
}
