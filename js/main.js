// 等待DOM加载完成
document.addEventListener('DOMContentLoaded', () => {
    // 获取元素
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');
    const backToTop = document.getElementById('back-to-top');

    // 移动端导航菜单切换
    navToggle.addEventListener('click', () => {
        navMenu.classList.toggle('active');
    });

    // 点击导航链接时关闭菜单
    document.querySelectorAll('.nav-menu a').forEach(link => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('active');
        });
    });

    // 滚动到顶部按钮
    window.addEventListener('scroll', () => {
        if (window.scrollY > 300) {
            backToTop.style.display = 'flex';
        } else {
            backToTop.style.display = 'none';
        }
    });

    backToTop.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    // 平滑滚动
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // 导航栏滚动效果
    let lastScroll = 0;
    const navbar = document.querySelector('.navbar');

    window.addEventListener('scroll', () => {
        const currentScroll = window.scrollY;

        if (currentScroll > lastScroll && currentScroll > 100) {
            // 向下滚动时隐藏导航栏
            navbar.style.transform = 'translateY(-100%)';
        } else {
            // 向上滚动时显示导航栏
            navbar.style.transform = 'translateY(0)';
        }

        lastScroll = currentScroll;
    });

    // 页面加载动画
    document.body.classList.add('loaded');
}); 