/**
 * Quantum Cryptography Tutorial - Main JavaScript
 * ==============================================
 * 
 * Provides enhanced user experience with smooth animations,
 * interactive elements, and modern UI interactions.
 */

class QuantumTutorialUI {
    constructor() {
        this.init();
    }

    init() {
        this.setupAnimations();
        this.setupInteractiveElements();
        this.setupNavigation();
        this.setupScrollEffects();
        this.setupLoadingStates();
    }

    setupAnimations() {
        // Intersection Observer for fade-in animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, observerOptions);

        // Observe elements for animation
        document.querySelectorAll('.course-card, .module-item, .feature-card, .content-section').forEach(el => {
            observer.observe(el);
        });
    }

    setupInteractiveElements() {
        // Enhanced button interactions
        document.querySelectorAll('.btn').forEach(btn => {
            btn.addEventListener('mouseenter', this.createRippleEffect.bind(this));
            btn.addEventListener('click', this.handleButtonClick.bind(this));
        });

        // Card hover effects
        document.querySelectorAll('.course-card, .module-item, .feature-card').forEach(card => {
            card.addEventListener('mouseenter', this.enhanceCardHover.bind(this));
            card.addEventListener('mouseleave', this.resetCardHover.bind(this));
        });

        // Interactive navigation
        document.querySelectorAll('.nav-menu a').forEach(link => {
            link.addEventListener('click', this.handleNavigation.bind(this));
        });
    }

    setupNavigation() {
        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(anchor.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Active navigation highlighting
        this.updateActiveNavigation();
        window.addEventListener('scroll', this.updateActiveNavigation.bind(this));
    }

    setupScrollEffects() {
        // Parallax effect for hero section
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const hero = document.querySelector('.hero-section');
            if (hero) {
                hero.style.transform = `translateY(${scrolled * 0.5}px)`;
            }
        });

        // Navbar background on scroll
        window.addEventListener('scroll', () => {
            const navbar = document.querySelector('.navbar');
            if (navbar) {
                if (window.scrollY > 100) {
                    navbar.classList.add('scrolled');
                } else {
                    navbar.classList.remove('scrolled');
                }
            }
        });
    }

    setupLoadingStates() {
        // Show loading state for API calls
        document.querySelectorAll('[data-loading]').forEach(element => {
            element.addEventListener('click', this.showLoadingState.bind(this));
        });
    }

    createRippleEffect(event) {
        const button = event.currentTarget;
        const ripple = document.createElement('span');
        const rect = button.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event.clientX - rect.left - size / 2;
        const y = event.clientY - rect.top - size / 2;

        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        ripple.classList.add('ripple');

        button.appendChild(ripple);

        setTimeout(() => {
            ripple.remove();
        }, 600);
    }

    handleButtonClick(event) {
        const button = event.currentTarget;
        
        // Add click animation
        button.classList.add('btn-clicked');
        setTimeout(() => {
            button.classList.remove('btn-clicked');
        }, 200);

        // Handle specific button actions
        if (button.classList.contains('btn-primary')) {
            this.trackEvent('primary_button_click', button.textContent.trim());
        }
    }

    enhanceCardHover(event) {
        const card = event.currentTarget;
        card.style.transform = 'translateY(-8px) scale(1.02)';
        card.style.boxShadow = '0 20px 40px rgba(0, 0, 0, 0.15)';
    }

    resetCardHover(event) {
        const card = event.currentTarget;
        card.style.transform = 'translateY(0) scale(1)';
        card.style.boxShadow = '';
    }

    handleNavigation(event) {
        const link = event.currentTarget;
        const navItems = document.querySelectorAll('.nav-menu a');
        
        navItems.forEach(item => item.classList.remove('active'));
        link.classList.add('active');
    }

    updateActiveNavigation() {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.nav-menu a[href^="#"]');
        
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (window.scrollY >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    }

    showLoadingState(event) {
        const element = event.currentTarget;
        const originalText = element.textContent;
        
        element.textContent = 'Loading...';
        element.disabled = true;
        element.classList.add('loading');

        // Simulate loading (replace with actual API call)
        setTimeout(() => {
            element.textContent = originalText;
            element.disabled = false;
            element.classList.remove('loading');
        }, 2000);
    }

    trackEvent(eventName, eventData) {
        // Analytics tracking (replace with your analytics service)
        console.log(`Event: ${eventName}`, eventData);
    }

    // Utility functions
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }

    animateCounter(element, target, duration = 1000) {
        let start = 0;
        const increment = target / (duration / 16);
        
        const timer = setInterval(() => {
            start += increment;
            element.textContent = Math.floor(start);
            
            if (start >= target) {
                element.textContent = target;
                clearInterval(timer);
            }
        }, 16);
    }
}

// Enhanced form handling
class FormHandler {
    constructor() {
        this.setupFormValidation();
    }

    setupFormValidation() {
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', this.handleFormSubmit.bind(this));
        });
    }

    handleFormSubmit(event) {
        event.preventDefault();
        const form = event.currentTarget;
        const formData = new FormData(form);
        
        // Add loading state
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Submitting...';
        }

        // Simulate form submission with error handling
        setTimeout(() => {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Submit';
            }
            const isValid = this.validateFormData(formData);
            if (isValid) {
                this.showSuccessMessage('Form submitted successfully!');
            } else {
                this.showErrorMessage('Form validation failed. Please check your inputs.');
            }
        }, 1500);
    }

    showSuccessMessage(message) {
        const ui = new QuantumTutorialUI();
        ui.showNotification(message, 'success');
    }
}

// Enhanced search functionality
class SearchHandler {
    constructor() {
        this.setupSearch();
    }

    setupSearch() {
        const searchInput = document.querySelector('.search-input');
        if (searchInput) {
            searchInput.addEventListener('input', this.handleSearch.bind(this));
        }
    }

    handleSearch(event) {
        const query = event.target.value.toLowerCase();
        const searchableElements = document.querySelectorAll('[data-searchable]');
        
        searchableElements.forEach(element => {
            const text = element.textContent.toLowerCase();
            if (text.includes(query)) {
                element.style.display = 'block';
                element.classList.add('search-highlight');
            } else {
                element.style.display = 'none';
                element.classList.remove('search-highlight');
            }
        });
    }
}

// Theme switcher
class ThemeSwitcher {
    constructor() {
        this.setupThemeToggle();
        this.loadThemePreference();
    }

    setupThemeToggle() {
        const themeToggle = document.querySelector('.theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', this.toggleTheme.bind(this));
        }
    }

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    }

    loadThemePreference() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
    }
}

// Initialize all components when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new QuantumTutorialUI();
    new FormHandler();
    new SearchHandler();
    new ThemeSwitcher();
    
    // Add CSS for animations
    const style = document.createElement('style');
    style.textContent = `
        .animate-in {
            animation: fadeInUp 0.6s ease-out forwards;
        }
        
        .btn-clicked {
            transform: scale(0.95);
        }
        
        .ripple {
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: scale(0);
            animation: ripple 0.6s linear;
            pointer-events: none;
        }
        
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            transform: translateX(100%);
            transition: transform 0.3s ease;
            z-index: 1000;
        }
        
        .notification.show {
            transform: translateX(0);
        }
        
        .notification-success {
            background: var(--success);
        }
        
        .notification-error {
            background: var(--error);
        }
        
        .notification-info {
            background: var(--info);
        }
        
        .search-highlight {
            background: rgba(14, 165, 233, 0.1);
            border-radius: 4px;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
        
        .navbar.scrolled {
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(10px);
        }
    `;
    document.head.appendChild(style);
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { QuantumTutorialUI, FormHandler, SearchHandler, ThemeSwitcher };
}
