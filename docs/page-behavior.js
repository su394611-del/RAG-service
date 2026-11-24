(() => {
  'use strict';
  for (const button of document.querySelectorAll('[data-copy-target]')) {
    button.addEventListener('click', async () => {
      const target = document.getElementById(button.dataset.copyTarget);
      if (!target) return;
      const text = target.textContent;
      const original = button.textContent;
      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(text);
        } else {
          const area = document.createElement('textarea');
          area.value = text; area.style.position = 'fixed'; area.style.opacity = '0';
          document.body.appendChild(area); area.select();
          if (!document.execCommand('copy')) throw new Error('Copy unavailable');
          area.remove();
        }
        button.textContent = 'Copied ✓';
      } catch (error) { button.textContent = 'Select & copy'; }
      window.setTimeout(() => { button.textContent = original; }, 1800);
    });
  }
  const mobileMenu = document.querySelector('.mobilemenu');
  if (mobileMenu) mobileMenu.querySelectorAll('a').forEach(a => a.addEventListener('click', () => { mobileMenu.open = false; }));
})();
